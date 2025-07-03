import io


def pil_image_to_bytes(pil_image, format="PNG"):

    """Converts a PIL Image to bytes in the specified format."""
    if pil_image is None:
        return None
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
