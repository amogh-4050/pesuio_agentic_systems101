import gradio as gr
from moviepy.editor import ImageSequenceClip
from PIL import Image
import os
import io
import tempfile
import torch
import base64
from transformers import CLIPProcessor, CLIPModel

# Define paths
IMAGE_BASE_PATH = "data/images/raw"
ORIENTATIONS = ["back", "follow", "front", "side", "top", "wing"]

# Image Processing Agent
class ImageProcessingAgent:
    def __init__(self):
        self.index_name = "suspect-identification"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embeddings = []

    def process_images(self, image_files):
        processed_count = 0
        for image_file in image_files:
            try:
                # Read image data
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Generate embedding
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embedding = self.model.get_image_features(**inputs).numpy()[0]
                    normalized_embedding = image_embedding / torch.norm(
                        torch.tensor(image_embedding)
                    )
                    
                # Convert image to base64 for storage
                buffered = io.BytesIO()
                image.save(buffered, format=image.format or "JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Store embeddings with metadata
                self.embeddings.append((image_file.name, normalized_embedding.tolist(), {"image_data": img_str}))
                processed_count += 1

            except Exception as e:
                print(f"Error processing image {image_file.name}: {str(e)}")

        return f"Processed {processed_count} images successfully."

# Video Creation Function
def create_video_from_images():
    image_paths = []
    for orientation in ORIENTATIONS:
        orientation_folder = os.path.join(IMAGE_BASE_PATH, orientation)
        if os.path.exists(orientation_folder):
            images = sorted(os.listdir(orientation_folder))
            for image_file in images:
                if image_file.endswith(".png"):
                    full_image_path = os.path.join(orientation_folder, image_file)
                    image_paths.append(full_image_path)

    if not image_paths:
        return "No images found to create a video."

    temp_video_file = tempfile.mktemp(suffix=".mp4")
    clip = ImageSequenceClip(image_paths, fps=2)
    clip.write_videofile(temp_video_file, codec="libx264")

    return temp_video_file

def create_gradio_demo():
    agent = ImageProcessingAgent()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simulation Video and Image Processing UI")

        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload Images",
                    file_types=[".png"]
                )
                upload_button = gr.Button("Process Images")
                status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("## Simulation Video")
                video_path = create_video_from_images()
                video_display = gr.Video(value=video_path if video_path else None, format="mp4")

                gr.Markdown("## Chatbot")
                chatbot_interface = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Type your message",
                        placeholder="Ask me anything...",
                        lines=2,
                        scale=4
                    )
                    submit_button = gr.Button("Submit", scale=1)
                clear_button = gr.Button("Clear")

        # Event Handlers
        upload_button.click(
            fn=agent.process_images,
            inputs=[file_output],
            outputs=[status_box]
        )

        submit_button.click(
            fn=lambda user_input, history: (history + [[user_input, "Response"]]), # Placeholder for chatbot response
            inputs=[msg, chatbot_interface],
            outputs=chatbot_interface,
        )

        clear_button.click(
            lambda: None,
            None,
            chatbot_interface,
            queue=False
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_demo()
    demo.launch(share=True)
