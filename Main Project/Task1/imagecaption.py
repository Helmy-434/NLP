import io
from typing import Union
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        device: str = None,):
        
        self.model_id = model_id
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = None
        self.model = None


    def load_model(self) -> None:
        if self.model is None or self.processor is None: # Lazy loading
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id
            ).to(self.device)
            self.model.eval()


    def _preprocess_image(
        self, image_input: Union[str, bytes, io.BytesIO, Image.Image]
    ) -> Image.Image:

        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, (str, bytes, io.BytesIO)):
            if isinstance(image_input, bytes):
                image_input = io.BytesIO(image_input)
            image = Image.open(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return image.convert("RGB")  # Transforms any image to RGB PIL format

    def generate_caption(
        self,
        image_input: Union[str, bytes, io.BytesIO, Image.Image],
        max_new_tokens: int = 50,
        num_beams: int = 3,) -> str:
    
       
        self.load_model()
        image = self._preprocess_image(image_input)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)# transform to tensors

        # Generate tokens without tracking gradients
        with torch.no_grad():
            output_tokens = self.model.generate( # contains the IDs of the words
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams, #beam search
                early_stopping=True,
            )

        caption = self.processor.decode(output_tokens[0], skip_special_tokens=True) #Transforms the IDs back to their original words
        return caption.strip()