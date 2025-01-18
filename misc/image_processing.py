import timeit
import matplotlib.pyplot as plt

def blur_img(img):
    top = img[:-2, 1:-1]
    left = img[1:-1, :-2]
    center = img[1:-1, 1:-1]
    bottom = img[2:, 1:-1]
    right = img[1:-1, 2:]
    return (top + left + center + bottom + right) / 5

def do_blurring(name: str):
    img = plt.imread(name)

    res = timeit.timeit(lambda: blur_img(img), number=100)
    print(res / 100)

    blurred = blur_img(img)
    for _ in range(10):
        blurred = blur_img(blurred)

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(blurred)

    plt.show()

if __name__ == '__main__':
    # do_blurring('resources/starbucks-logo.png')