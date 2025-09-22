# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
DATASET="${1:-multigrid}"                                   # multigrid, procgen, sports, nuplan, gr00t, egtea
LOAD_CHECKPOINT="${2:-}"
ONLY_EVAL_TEST="${3:-false}"
echo "Using dataset: $DATASET"

DATA_PATH_BASE="/scratch/cluster/zzwang_new/flam_data"
ENV_NAME="flam"
LEARNING_RATE=5e-5
BATCH_SIZE=128
TRAINING_STEPS=50000
EVAL_FREQ=2500
EVAL_SKIP_STEPS=0
LPIPS_WEIGHT=1.0
EVAL_FID=false
EVAL_FVD=false
NUM_WORKERS=4
PREFETCH_FACTOR=2
SUB_TRAJ_LEN=2
VALIDATION_DATASET_PERCENTAGE=1.0
N_EVAL_STEPS=10
N_VALID_EVAL_SAMPLE_IMAGES=10
N_TEST_EVAL_SAMPLE_IMAGES=25


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
WM_SCALE=24
POLICY_IMPALA_SCALE=4

# Vector Quantization (VQ) Settings
VQ_ENABLED=true
VQ_NUM_CODEBOOKS=2
VQ_NUM_DISCRETE_LATENTS=4
VQ_EMB_DIM=16
VQ_NUM_EMBS=64
VQ_COMMITMENT_COST=0.05
VQ_DECAY=0.999
VQ_WARMUP_STEPS=0

# =============================================================================
# ENCODER/DECODER CONFIGURATION
# =============================================================================

# WM MagViT2 Encoder Defaults
WM_ENCODER_MAGVIT2_CH=128
WM_ENCODER_MAGVIT2_NUM_RES_BLOCKS=4
WM_ENCODER_MAGVIT2_CH_MULT=[1,1,2,2,4]
WM_ENCODER_MAGVIT2_Z_CHANNELS=8

# WM Impala Encoder Defaults
WM_ENCODER_IMPALA_CH=64
WM_ENCODER_IMPALA_NUM_RES_BLOCKS=4
WM_ENCODER_IMPALA_CH_MULT=[1,2,2,4]
WM_ENCODER_IMPALA_Z_CHANNELS=8

# WM MagViT2 Decoder Defaults
WM_DECODER_MAGVIT2_CH=128
WM_DECODER_MAGVIT2_NUM_RES_BLOCKS=4
WM_DECODER_MAGVIT2_CH_MULT=[1,1,2,2,4]
WM_DECODER_MAGVIT2_Z_CHANNELS=8

# WM Lapo Decoder Defaults
WM_DECODER_LAPO_CH=48
WM_DECODER_LAPO_CH_MULT=[16,8,4,2]
WM_DECODER_LAPO_Z_CHANNELS=8

# IDM MagViT2 Encoder Defaults
IDM_ENCODER_MAGVIT2_CH=128
IDM_ENCODER_MAGVIT2_NUM_RES_BLOCKS=2
IDM_ENCODER_MAGVIT2_CH_MULT=[1,1,2,2,4]
IDM_ENCODER_MAGVIT2_Z_CHANNELS=8

# IDM Impala Encoder Defaults
IDM_ENCODER_IMPALA_CH=64
IDM_ENCODER_IMPALA_NUM_RES_BLOCKS=4
IDM_ENCODER_IMPALA_CH_MULT=[4,2,2,1]
IDM_ENCODER_IMPALA_Z_CHANNELS=8


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

if [ "$DATASET" = "multigrid" ]; then
    EXP_NAME="multigrid"
    TRAINING_STEPS=50000

    # Model
    VQ_COMMITMENT_COST=0.01
    VQ_DECAY=0.7
    VQ_WARMUP_STEPS=100

    # Data
    DATA_PATH="$DATA_PATH_BASE/multigrid/"
    IMAGE_SIZE=128
    FRAME_SKIP=1
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="empty-agent_4-v0/data/main_data.hdf5"     # v0: train (1M), v1: valid (10k), v2: test (100k)
    VALID_DATASET_ID="empty-agent_4-v1/data/main_data.hdf5"
    TEST_DATASET_ID="empty-agent_4-v2/data/main_data.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="impala"
    WM_DECODER_TYPE="lapo"
    IDM_ENCODER_TYPE="impala"
elif [ "$DATASET" = "procgen" ]; then
    EXP_NAME="procgen"
    BATCH_SIZE=32
    PROCGEN_USE_BACKGROUND=False
    TRAINING_STEPS=100000
    EVAL_FREQ=5000

    # Model
    LEARNING_RATE=5e-5
    VQ_COMMITMENT_COST=0.1
    VQ_DECAY=0.7
    VQ_WARMUP_STEPS=100

    # Data
    DATA_PATH="$DATA_PATH_BASE/procgen/"
    IMAGE_SIZE=224
    FRAME_SKIP=1
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="\\*background_${PROCGEN_USE_BACKGROUND}-v0/data/main_data.hdf5"
    VALID_DATASET_ID="\\*background_${PROCGEN_USE_BACKGROUND}-v1/data/main_data.hdf5"
    TEST_DATASET_ID="\\*background_${PROCGEN_USE_BACKGROUND}-v2/data/main_data.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="impala"
    WM_DECODER_TYPE="lapo"
    IDM_ENCODER_TYPE="impala"
elif [ "$DATASET" = "sports" ]; then
    EXP_NAME="sports"
    LEARNING_RATE=5e-5
    BATCH_SIZE=8
    TRAINING_STEPS=50000
    EVAL_FREQ=5000
    VALIDATION_DATASET_PERCENTAGE=0.12
    EVAL_SKIP_STEPS=15001

    # Model
    VQ_COMMITMENT_COST=0.15
    VQ_DECAY=0.7

    # Data
    DATA_PATH="$DATA_PATH_BASE/"
    IMAGE_SIZE=224
    FRAME_SKIP=3
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="basketball/train.hdf5,sports_mot/train.hdf5,tennis_play_video_generation/train.hdf5,tenniset/train.hdf5,volleyball/train.hdf5"
    VALID_DATASET_ID="basketball/valid.hdf5,sports_mot/valid.hdf5,tennis_play_video_generation/valid.hdf5,tenniset/valid.hdf5,volleyball/valid.hdf5"
    TEST_DATASET_ID="basketball/test.hdf5,sports_mot/test.hdf5,tennis_play_video_generation/test.hdf5,tenniset/test.hdf5,volleyball/test.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="magvit2"
    WM_DECODER_TYPE="magvit2"
    IDM_ENCODER_TYPE="magvit2"
elif [ "$DATASET" = "nuplan" ]; then
    EXP_NAME="nuplan"
    LEARNING_RATE=5e-5
    BATCH_SIZE=8
    TRAINING_STEPS=50000
    EVAL_FREQ=5000
    VALIDATION_DATASET_PERCENTAGE=0.15
    EVAL_SKIP_STEPS=10001

    # Model
    VQ_COMMITMENT_COST=0.15
    VQ_DECAY=0.7

    # Data
    DATA_PATH="$DATA_PATH_BASE/"
    IMAGE_SIZE=224
    FRAME_SKIP=2
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="nuplan_mini/train_CAM_F0.hdf5,nuplan_val/train_CAM_F0.hdf5"
    VALID_DATASET_ID="nuplan_mini/valid_CAM_F0.hdf5,nuplan_val/valid_CAM_F0.hdf5"
    TEST_DATASET_ID="nuplan_mini/test_CAM_F0.hdf5,nuplan_val/test_CAM_F0.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="magvit2"
    WM_DECODER_TYPE="magvit2"
    IDM_ENCODER_TYPE="magvit2"
elif [ "$DATASET" = "gr00t" ]; then
    EXP_NAME="gr00t"
    BATCH_SIZE=8

    # Model
    LEARNING_RATE=5e-5
    VQ_COMMITMENT_COST=0.15
    VQ_DECAY=0.7

    # Data
    DATA_PATH="$DATA_PATH_BASE/"
    IMAGE_SIZE=224
    FRAME_SKIP=2
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="gr00t/train.hdf5"
    VALID_DATASET_ID="gr00t/valid.hdf5"
    TEST_DATASET_ID="gr00t/test.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="magvit2"
    WM_DECODER_TYPE="magvit2"
    IDM_ENCODER_TYPE="magvit2"
elif [ "$DATASET" = "egtea" ]; then
    EXP_NAME="egtea"
    BATCH_SIZE=8
    TRAINING_STEPS=50000
    EVAL_FREQ=5000
    VALIDATION_DATASET_PERCENTAGE=0.05
    EVAL_SKIP_STEPS=15001

    # Model
    LEARNING_RATE=5e-5
    VQ_COMMITMENT_COST=0.15
    VQ_DECAY=0.7

    # Data
    DATA_PATH="$DATA_PATH_BASE/"
    IMAGE_SIZE=224
    FRAME_SKIP=1
    ITERATE_FRAME_BETWEEN_SKIP=true
    TRAIN_DATASET_ID="egtea/train.hdf5"
    VALID_DATASET_ID="egtea/valid.hdf5"
    TEST_DATASET_ID="egtea/test.hdf5"

    # Encoder/Decoder settings
    WM_ENCODER_TYPE="magvit2"
    WM_DECODER_TYPE="magvit2"
    IDM_ENCODER_TYPE="magvit2"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi


# =============================================================================
# RUN TRAINING
# =============================================================================
python -u stage1_idm.py \
    env_name="$ENV_NAME" \
    exp_name="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)" \
    image_size=$IMAGE_SIZE \
    sub_traj_len=$SUB_TRAJ_LEN \
    data.path="$DATA_PATH" \
    data.train_fname="$TRAIN_DATASET_ID" \
    data.valid_fname="$VALID_DATASET_ID" \
    data.test_fname="$TEST_DATASET_ID" \
    data.frame_skip=$FRAME_SKIP \
    data.iterate_frame_between_skip=$ITERATE_FRAME_BETWEEN_SKIP \
    data.num_workers=$NUM_WORKERS \
    data.prefetch_factor=$PREFETCH_FACTOR \
    model.wm_scale=$WM_SCALE \
    model.policy_impala_scale=$POLICY_IMPALA_SCALE \
    model.vq.enabled=$VQ_ENABLED \
    model.vq.num_codebooks=$VQ_NUM_CODEBOOKS \
    model.vq.num_discrete_latents=$VQ_NUM_DISCRETE_LATENTS \
    model.vq.emb_dim=$VQ_EMB_DIM \
    model.vq.num_embs=$VQ_NUM_EMBS \
    model.vq.commitment_cost=$VQ_COMMITMENT_COST \
    model.vq.decay=$VQ_DECAY \
    model.vq.warmup_steps=$VQ_WARMUP_STEPS \
    model.idm_encoder.encoder_type="$IDM_ENCODER_TYPE" \
    model.idm_encoder.encoder_all.magvit2.ch=$IDM_ENCODER_MAGVIT2_CH \
    model.idm_encoder.encoder_all.magvit2.num_res_blocks=$IDM_ENCODER_MAGVIT2_NUM_RES_BLOCKS \
    model.idm_encoder.encoder_all.magvit2.ch_mult=$IDM_ENCODER_MAGVIT2_CH_MULT \
    model.idm_encoder.encoder_all.magvit2.z_channels=$IDM_ENCODER_MAGVIT2_Z_CHANNELS \
    model.idm_encoder.encoder_all.impala.ch=$IDM_ENCODER_IMPALA_CH \
    model.idm_encoder.encoder_all.impala.num_res_blocks=$IDM_ENCODER_IMPALA_NUM_RES_BLOCKS \
    model.idm_encoder.encoder_all.impala.ch_mult=$IDM_ENCODER_IMPALA_CH_MULT \
    model.idm_encoder.encoder_all.impala.z_channels=$IDM_ENCODER_IMPALA_Z_CHANNELS \
    model.wm_encdec.encoder_type="$WM_ENCODER_TYPE" \
    model.wm_encdec.encoder_all.magvit2.ch=$WM_ENCODER_MAGVIT2_CH \
    model.wm_encdec.encoder_all.magvit2.num_res_blocks=$WM_ENCODER_MAGVIT2_NUM_RES_BLOCKS \
    model.wm_encdec.encoder_all.magvit2.ch_mult=$WM_ENCODER_MAGVIT2_CH_MULT \
    model.wm_encdec.encoder_all.magvit2.z_channels=$WM_ENCODER_MAGVIT2_Z_CHANNELS \
    model.wm_encdec.encoder_all.impala.ch=$WM_ENCODER_IMPALA_CH \
    model.wm_encdec.encoder_all.impala.num_res_blocks=$WM_ENCODER_IMPALA_NUM_RES_BLOCKS \
    model.wm_encdec.encoder_all.impala.ch_mult=$WM_ENCODER_IMPALA_CH_MULT \
    model.wm_encdec.encoder_all.impala.z_channels=$WM_ENCODER_IMPALA_Z_CHANNELS \
    model.wm_encdec.decoder_type="$WM_DECODER_TYPE" \
    model.wm_encdec.decoder_all.magvit2.ch=$WM_DECODER_MAGVIT2_CH \
    model.wm_encdec.decoder_all.magvit2.num_res_blocks=$WM_DECODER_MAGVIT2_NUM_RES_BLOCKS \
    model.wm_encdec.decoder_all.magvit2.ch_mult=$WM_DECODER_MAGVIT2_CH_MULT \
    model.wm_encdec.decoder_all.magvit2.z_channels=$WM_DECODER_MAGVIT2_Z_CHANNELS \
    model.wm_encdec.decoder_all.lapo.ch=$WM_DECODER_LAPO_CH \
    model.wm_encdec.decoder_all.lapo.ch_mult=$WM_DECODER_LAPO_CH_MULT \
    model.wm_encdec.decoder_all.lapo.z_channels=$WM_DECODER_LAPO_Z_CHANNELS \
    stage1.lr=$LEARNING_RATE \
    stage1.bs=$BATCH_SIZE \
    stage1.steps=$TRAINING_STEPS \
    stage1.only_eval_test=$ONLY_EVAL_TEST \
    stage1.load_checkpoint="$LOAD_CHECKPOINT" \
    stage1.eval_freq=$EVAL_FREQ \
    stage1.eval_skip_steps=$EVAL_SKIP_STEPS \
    stage1.valid_dataset_percentage=$VALIDATION_DATASET_PERCENTAGE \
    stage1.n_eval_steps=$N_EVAL_STEPS \
    stage1.n_valid_eval_sample_images=$N_VALID_EVAL_SAMPLE_IMAGES \
    stage1.n_test_eval_sample_images=$N_TEST_EVAL_SAMPLE_IMAGES \
    stage1.image_loss.lpips_weight=$LPIPS_WEIGHT \
    stage1.image_loss.eval_fid=$EVAL_FID \
    stage1.image_loss.eval_fvd=$EVAL_FVD
