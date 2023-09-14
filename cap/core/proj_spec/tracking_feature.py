import json


def get_rcnn_tracking_feature_desc(feature_size, class_name):
    assert isinstance(feature_size, (list, tuple))
    assert len(feature_size) == 4
    for i in feature_size:
        assert isinstance(i, int), type(i)

    desc = {
        "task": "frcnn_tracking_feature",
        "size": feature_size,
        "class_name": class_name,
    }

    return json.dumps(desc)
