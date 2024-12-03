import urllib.request
import urllib


def download_image(url, filename):
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)


def download_dog_image():
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = "dog.jpg"
    download_image(url, filename)
