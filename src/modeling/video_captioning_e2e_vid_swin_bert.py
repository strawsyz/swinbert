import torch
from fairscale.nn.misc import checkpoint_wrapper
import cv2
import random


class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        # RVOS_mask = kwargs['RVOS_mask']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        #  print("images_shape", images.shape)  # 3,64,224,224
        import numpy as np
        # prediction_path = r"/workspace/MTTR/prediction-all.npy"
        # if RVOS_mask is not None:
        #     prediction = RVOS_mask
        #     mask = cv2.resize(prediction, dsize=(7, 7))
        #
        #     def normalize_mask(mask, min, max=1):
        #         assert max != 0
        #         return mask * (max - min) + min
        #
        #     mask = normalize_mask(mask, 0.1)

        # mask = torch.from_numpy(mask).float().cuda()
        # images[:, :, 16:48] = images[:, :, 16:48] * mask
        # images[:, :, 26:32] = images[:, :, 26:32] * mask
        # images[:, :, 32:38] = images[:, :, 32:38] * mask
        # images[:, :, 38:45] = images[:, :, 38:45] * mask
        # images = images * mask
        vid_feats = self.swin(images)
        # print("vid_feats", vid_feats.shape)  # [1, 1024, 32, 7, 7]
        # if RVOS_mask is not None:
        #     # vid_feats[:, 26:32] = vid_feats[:, 26:32] * torch.from_numpy(mask).cuda()
        #     # vid_feats[:, 32:38] = vid_feats[:, 32:38] * torch.from_numpy(mask).cuda()
        #     vid_feats[:, 26:38] = vid_feats[:, 26:38] * torch.from_numpy(mask).cuda()
        #     # vid_feats[:, 38:45] = vid_feats[:, 38:45] * torch.from_numpy(mask).cuda()
        #     vid_feats = vid_feats * torch.from_numpy(mask).float().cuda()

        if self.use_grid_feat == True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)
        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        #         print("vid_feats-shape", vid_feats.shape) # 1, 1568, 512
        #         print("self.learn_mask_enabled", self.learn_mask_enabled)  # True
        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            #             print("kwargs['attention_mask']", kwargs['attention_mask'])
            #             print("shape", kwargs['attention_mask'].shape)  # 1588,1588
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask) * learn_att
            learn_att = diag_mask + video_attention
            #             print("vid_att_len",vid_att_len)  # 1568
            #             print("learn_att", learn_att.shape)  #  1568, 1568
            #             print("diag_mask", diag_mask,diag_mask.shape)  # 1568, 1568
            #             print("video_attention", video_attention, video_attention.shape)  # 1568, 1568
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att >= 0.5) * 1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            #             print("learn_att", learn_att, learn_att.shape)  # 1568, 1568
            # print("kwargs['attention_mask']", kwargs['attention_mask'].shape)
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs = self.trans_encoder(*args, **kwargs)
        # print(outputs[0])
        # print(outputs[0].shape)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)
            outputs = outputs + (loss_sparsity,)
        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length / pretrained_num_tokens

        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        scale_factor = int(self.max_img_seq_length / pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None, None, :, :].double())[0, 0, :, :].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length * self.max_img_seq_length, 1)

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad = not freeze


class CustomVideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(CustomVideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False  # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        self.num_classes = 2
        self.classifer = torch.nn.Linear(20, self.num_classes)

        if self.learn_mask_enabled == True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length * args.max_img_seq_length, 1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        RVOS_mask = kwargs['RVOS_mask']
        # print(RVOS_mask)
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        #  print("images_shape", images.shape)  # 3,64,224,224
        import numpy as np
        # prediction_path = r"/workspace/MTTR/prediction-all.npy"
        prediction = RVOS_mask
        mask = cv2.resize(prediction, dsize=(7, 7))

        def normalize_mask(mask, min, max=1):
            assert max != 0
            return mask * (max - min) + min

        mask = normalize_mask(mask, 0.1)

        # mask = torch.from_numpy(mask).float().cuda()
        # images[:, :, 16:48] = images[:, :, 16:48] * mask
        # images[:, :, 26:32] = images[:, :, 26:32] * mask
        # images[:, :, 32:38] = images[:, :, 32:38] * mask
        # images[:, :, 38:45] = images[:, :, 38:45] * mask
        # images = images * mask
        vid_feats = self.swin(images)
        # print("vid_feats", vid_feats.shape)  # [1, 1024, 32, 7, 7]

        # vid_feats[:, 26:32] = vid_feats[:, 26:32] * torch.from_numpy(mask).cuda()
        # vid_feats[:, 32:38] = vid_feats[:, 32:38] * torch.from_numpy(mask).cuda()
        vid_feats[:, 26:38] = vid_feats[:, 26:38] * torch.from_numpy(mask).cuda()
        # vid_feats[:, 38:45] = vid_feats[:, 38:45] * torch.from_numpy(mask).cuda()
        vid_feats = vid_feats * torch.from_numpy(mask).float().cuda()

        if self.use_grid_feat == True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)
        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        #         print("vid_feats-shape", vid_feats.shape) # 1, 1568, 512
        #         print("self.learn_mask_enabled", self.learn_mask_enabled)  # True
        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            #             print("kwargs['attention_mask']", kwargs['attention_mask'])
            #             print("shape", kwargs['attention_mask'].shape)  # 1588,1588
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask) * learn_att
            learn_att = diag_mask + video_attention
            #             print("vid_att_len",vid_att_len)  # 1568
            #             print("learn_att", learn_att.shape)  #  1568, 1568
            #             print("diag_mask", diag_mask,diag_mask.shape)  # 1568, 1568
            #             print("video_attention", video_attention, video_attention.shape)  # 1568, 1568
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att >= 0.5) * 1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            #             print("learn_att", learn_att, learn_att.shape)  # 1568, 1568
            print("kwargs['attention_mask']", kwargs['attention_mask'].shape)
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs = self.trans_encoder(*args, **kwargs)
        cap_feature = outputs[0]
        outputs = self.classifer(cap_feature)
        # if self.learn_mask_enabled:
        #     loss_sparsity = self.get_loss_sparsity(video_attention)
        #     outputs = outputs + (loss_sparsity,)
        return outputs

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length / pretrained_num_tokens

        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        scale_factor = int(self.max_img_seq_length / pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None, None, :, :].double())[0, 0, :, :].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length * self.max_img_seq_length, 1)

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad = not freeze


if __name__ == '__main__':
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score, roc_curve
    from matplotlib import pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity


    # def sentence_similarity(s1: str, s2: str, model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2')):
    #     '''
    #
    #     Args:
    #         s1: such as, That is a happy person
    #         s2: such as, That is a happy dog
    #         model:
    #
    #     Returns:
    #         score
    #     '''
    #     embeddings = model.encode([s1, s2])
    #     res = cosine_similarity([embeddings[0]], [embeddings[1]])
    #     return res

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    s1 = s2 = "asdasd"
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # score = sentence_similarity(s1, s2, model=model)
