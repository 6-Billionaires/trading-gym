from datetime import datetime

def get_timestamp():
    """
    get current time "YYYYMMDD24HMMSS"
    :return:
    """
    return "{:%Y%m%d%H%M%S}".format(datetime.now())

