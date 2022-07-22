

intervals= [[1,3],[1,4],[1,5],[1,6]]
intervals.sort(key = lambda x:(x[0] , -x[1]))
print(intervals)