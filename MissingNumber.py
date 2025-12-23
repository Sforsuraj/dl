n=int(input())
l=list(map(int,input().split()))
s=sum(l)
total=(n*(n+1))//2
print(total-s)