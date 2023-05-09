from functools import reduce

dataset_embed_dim = {"vctk": 256, "chs": 512}
dataset_spk_num = {"vctk": 109, "chs": 299}

class hparams:

    dataset = "chs"
    # Audio:
    num_mels = 80
    num_linear = 513
    embed_dim = dataset_embed_dim[dataset]
    num_freq = 1025
    sample_rate = 16000
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20

    # Dataset
    batchsize = 4
    spk_num = dataset_spk_num[dataset]

    # Training
    model_type = 'autovc-v2'  # option 'autovc' 'autovc-v2'
    use_cuda = True
    epoch_num = 100000
    max_step = 12000
    #原来max_step = 600   200
    #原来的log_interval = 100
    log_interval = 200
    #checkpoint_interval = 30000
    #原来checkpoint_interval = 200
    checkpoint_interval = 1200
    #原来的learning_rate = 0.0001
    learning_rate = 0.001
    learning_rate2 = 0.0001
    #lr_decay_power = 0.5
    warmup_step = 4000
    #warmup_step = 2000
    dropout_rate = 0.1
    clip_thresh = 10

    dim_neck = 16
    dim_emb = 256
    dim_pre = 512

    #loss_ctc = 0.001
    loss_ctc = 0
    
    freq = 12

    # gst options
    E = 256
    # reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3, 3]
    ref_enc_strides = [2, 2]
    ref_enc_pad = [1, 1]
    ref_enc_gru_size = E // 2
    # style token layer
    token_num = 10
    # token_emb_size = 256
    num_heads = 8
    # multihead_attn_num_unit = 256
    # style_att_type = 'mlp_attention'
    # attn_normalize = True
    
    # use VQ
    use_EMA_vq = False
    loss_weight_vq = 1
    #vq_embed_dim = 128
    vq_embed_dim = 64
    vq_num_embed = 40
    commitment_cost = 0.25
    decay = 0.99    
