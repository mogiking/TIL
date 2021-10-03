n = 5
m = 16
listdduk = [12,23,17,13,16]

start = 0
end = max(listdduk)
result = 0

while (start<=end):
    total = 0
    mid = (start+end)//2
    for dduk in listdduk:
        if dduk > mid:
            total += dduk - mid
    if total < m:
        end = mid - 1
    else:
        result = mid
        start = mid + 1

print(result)
'''
************
*********
**********
*******  
'''