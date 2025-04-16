import torch
import torch.nn.functional as F

from custom_nodes.ComfyUI_SAM2_.sam2.modeling.sam2_base import SAM2Base, NO_OBJ_SCORE
from .sam2_video_predictor import SAM2VideoPredictor
from ..sam2.modeling.sam.prompt_encoder import PromptEncoder
from ..sam2.modeling.sam.mask_decoder import MaskDecoder


class CustomVideoPredictor(SAM2VideoPredictor):
    """
    A custom VideoPredictor that overrides the image size at construction time
    and rebuilds the relevant SAM heads (prompt encoder, mask decoder, etc.).
    """
    def __init__(self,
                 image_encoder,
                 memory_attention,
                 memory_encoder,
                 image_size=1024,
                 **kwargs):  # Forward everything else to base
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            image_size=image_size,
            **kwargs
        )
        # After the parent's constructor sets self.image_size, rebuild the heads
        self._rebuild_sam_heads()
        self._resize_positional_embeddings()

    def _rebuild_sam_heads(self):
        """
        Recreate the PromptEncoder and MaskDecoder for the new image_size.
        """
        print("DEBUG: TRYING TO REBUILD SAM HEAD")
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # Re-init the prompt encoder
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size,
                                  self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

        # # Re-init the mask decoder (tweak num_multimask_outputs etc. as you need)
        # self.sam_mask_decoder = MaskDecoder(
        #     num_multimask_outputs=3 if self.multimask_output_in_sam else 1,
        #     transformer_dim=self.sam_prompt_embed_dim,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        #     use_high_res_features=self.use_high_res_features_in_sam,
        #     iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
        #     pred_obj_scores=self.pred_obj_scores,
        #     pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        #     use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
        #     **(self.sam_mask_decoder_extra_args or {}),
        # )

        # If you have an obj_ptr_proj or no_obj_ptr, they're created in SAM2Base __init__,
        # so no special re-init is needed unless you want to modify them.

    def _resize_positional_embeddings(self):
        """
        If the PromptEncoder uses a dense positional embedding that depends on image size,
        resize it via bilinear interpolation.
        """
        encoder = self.sam_prompt_encoder
        if hasattr(encoder, "get_dense_pe") and hasattr(encoder, "set_dense_pe"):
            old_pe = encoder.get_dense_pe()
            new_H = new_W = self.sam_image_embedding_size
            if old_pe.shape[-2:] != (new_H, new_W):
                print(f"[INFO] Resizing pos encoding from {old_pe.shape[-2:]} to {(new_H, new_W)}")
                new_pe = F.interpolate(old_pe, size=(new_H, new_W),
                                       mode="bilinear", align_corners=False)
                encoder.set_dense_pe(new_pe)
