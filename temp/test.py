def quick_sort(array, low=None, high=None):
    if low is None:
        low = 0
    if high is None:
        high = len(array) - 1

    if low < high:
        partition_index = partition(array, low, high)
        quick_sort(array, low, partition_index - 1)
        quick_sort(array, partition_index + 1, high)

def partition(array, low, high):
    pivot = array[high]
    i = low - 1

    for j in range(low, high):
        if array[j] < pivot:
            i += 1
            array[i], array[j] = array[j], array[i]

    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

if __name__ == "__main__":
# 测试用例
    array = [5, 3, 7, 1, 9, 2, 4, 6, 8]
    quick_sort(array)
    print(array)