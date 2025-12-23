n=int(input())
l=list(map(int,input().split()))
moves=0
for i in range(1,n):
    if l[i]<=l[i-1]:
        moves+=l[i-1]-l[i]
        l[i]=l[i-1]
print(moves)