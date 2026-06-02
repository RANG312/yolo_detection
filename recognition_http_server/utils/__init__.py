__all__ = ["prepare_image"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "prepare_image":
        from recognition_http_server.utils.image_io import prepare_image

        return prepare_image
    raise AttributeError(name)
