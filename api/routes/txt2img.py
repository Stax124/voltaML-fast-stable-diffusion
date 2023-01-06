from fastapi import APIRouter, HTTPException

from api.shared import state
from core import queue
from core.types import Txt2ImgQueueEntry
from core.utils import convert_image_to_base64

router = APIRouter()


@router.post("/interupt")
async def stop():
    state.interrupt = True
    return {"message": "Interupted"}


@router.post("/generate")
async def txt2img_job(job: Txt2ImgQueueEntry):
    # Create directory to save images if it does not exist

    if job.backend == "PyTorch":
        images, time = await queue.add_job(job)
    elif job.backend == "TensorRT":
        from volta_accelerate import infer_trt
        infer_trt(
            saving_path="static/output/"+str(job.data.id),
            model=job.model.value,
            prompt=job.data.prompt,
            neg_prompt=job.data.negative_prompt,
            img_height=job.data.height,
            img_width=job.data.width,
            num_inference_steps=job.data.steps,
            guidance_scale=job.data.guidance_scale,
            num_images_per_prompt=job.data.batch_size,
            seed=job.data.seed,
        )
        images, time = list(), 0
        # infer_trt()
    else:
        raise HTTPException(status_code=400, detail="Invalid backend")

    return {"message": "Job completed", 
            "time": time, 
            "images": [convert_image_to_base64(i) for i in images]}
