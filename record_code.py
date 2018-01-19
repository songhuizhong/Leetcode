# -*- coding: iso-8859-15 -*-
# Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0?
# Find all unique triplets in the array which gives the sum of zero.
# For example, given array S = [-1, 0, 1, 2, -1, -4],
#
# A solution set is:
# [
#   [-1, 0, 1],
#   [-1, -1, 2]
# ]
# My answer
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]

        """
        mylist = []
        nums = sorted(nums)
        for i in range(len(nums) - 2):
            if i == 0 or (i > 0 and nums[i] != nums[i - 1]):
                low = i + 1
                high = len(nums) - 1
                mysum = 0 - nums[i]
                while low < high:
                    if nums[low] + nums[high] == mysum:
                        mylist.append([nums[i], nums[low], nums[high]])
                        while low < len(nums) - 1 and nums[low] == nums[low + 1]: low += 1
                        while 0 < high and nums[high] == nums[high - 1]: high -= 1
                        low += 1
                        high -= 1
                    elif nums[low] + nums[high] > mysum:
                        high -= 1
                    else:
                        low += 1
        return mylist


# def findrecursive(self,nums,length,newlist,mylist):
#         print nums
#         if length==3 and sum(newlist)==0:
#                 newlist.sort()
#                 print newlist
#                 if newlist not in mylist:
#                     otherlist=newlist[:]#这里一定要注意如果没有另外赋值的话，mylist中的元素就会跟着newlist一起变化
#                     #因为往里面传的是reference
#                     mylist.append(otherlist)
#                 #     print mylist
#         if length==3 and sum(newlist)!=0:
#             return
#         for i in range(len(nums)):
#             newlist.append(nums[i])
#             length+=1
#             self.findrecursive(nums[:i]+nums[i+1:],length,newlist,mylist)
#             newlist.pop()
#             length-=1
#
# good answer
# *******************************
# class Solution(object):
#     def threeSum(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         counter = {}
#         for num in nums:
#             if num not in counter:
#                 counter[num] = 0
#             counter[num] += 1
#
#         if 0 in counter and counter[0] > 2:
#             rst = [[0, 0, 0]]
#         else:
#             rst = []
#
#         uniques = counter.keys()  # 它使用hash table来过滤重复数字
#
#         pos = sorted(p for p in uniques if p > 0)  # 我只利用了将数组排列这一特征
#         neg = sorted(n for n in uniques if n < 0)
#
#         # 我也采取了分组(正数和负数)这一特征
#         for p in pos:
#             for n in neg:
#                 inverse = -(p + n)  # 通过设置sub-goal来设置
#                 if inverse in counter:
#                     if inverse == p and counter[p] > 1:
#                         rst.append([n, p, p])
#                     elif inverse == n and counter[n] > 1:
#                         rst.append([n, n, p])
#                     elif n < inverse < p or inverse == 0:
#                         rst.append([n, inverse, p])
#         return rst
# *****************************
# Given a string, find the length of the longest substring without repeating characters.

# Examples:
#
# Given "abcabcbb", the answer is "abc", which the length is 3.
#
# Given "bbbbb", the answer is "b", with the length of 1.
#
# Given "pwwkew", the answer is "wke", with the length of 3.
#  Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == '':
            return 0
        dict = {}
        for i in range(256):
            dict[chr(i)] = -1
        previous = -1
        current = -1
        maxLength = -float('inf')
        for i in range(len(s)):
            if dict[s[i]] != -1:
                current = dict[s[i]]
                dict[s[i]] = i
            else:
                dict[s[i]] = i
            previous = max(previous, current)
            if i - previous > maxLength:
                maxLength = i - previous
        return maxLength


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        indexes = {}
        longest = 0
        last_repeating = -1
        for i, c in enumerate(s):
            if c in indexes and last_repeating < indexes[c]:

            # if c in indexes :
                last_repeating = indexes[c]
            if i - last_repeating > longest:
                longest = i - last_repeating
            indexes[c] = i
            print

        return longest

#
# 第三题：There are two sorted arrays nums1 and nums2 of size m and n respectively.
# Find the median of the two sorted arrays. The overall run time
# complexity should be O(log (m+n)).
#nums1 = [1, 3]
# nums2 = [2]
#
# The median is 2.0
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        newlist=nums1+nums2
        newlist=sorted(newlist)
        list_length=len(newlist)
        if list_length&1==0:
            return ((newlist[list_length//2-1]+newlist[list_length//2])+0.0)/2.0
        else:
            return newlist[list_length//2]
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """

        """
        def findKth(A, m, B, n, k):

            if m > n:
                return findKth(B, n, A, m, k)
            elif m == 0:
                return B[k-1]
            elif k == 1:
                return min(A[0], B[0])

            # divide k into two parts
            pA = min(k/2, m)
            pB = k - pA

            if A[pA-1] < B[pB-1]:
                return findKth(A[pA:], m-pA, B, n, k-pA)
            elif A[pA-1] > B[pB-1]:
                return findKth(A, m, B[pB:], n-pB, k-pB)
            else:
                return A[pA-1]

        tmp = len(nums1)+len(nums2)
        if tmp % 2 == 0:
            return (findKth(nums1, len(nums1), nums2, len(nums2), tmp/2) +
                    findKth(nums1, len(nums1), nums2, len(nums2), tmp/2 + 1))/2.0
        else:
            return findKth(nums1, len(nums1), nums2, len(nums2), tmp/2 + 1)
        """

        m, n = len(nums1), len(nums2)
        if m > n:
            nums2, nums1, n, m = nums1, nums2, m, n

        if m == 0:
            if n % 2 == 0:
                return (nums2[n / 2 - 1] + nums2[n / 2]) / 2.0
            else:
                return nums2[n / 2]

        half = (m + n + 1) / 2
        left, right = 0, m
        while left <= right:
            i = (left + right) / 2
            j = half - i
            if i < m and nums2[j - 1] > nums1[i]:
                left = i + 1
            elif i > 0 and nums1[i - 1] > nums2[j]:
                right = i - 1
            else:
                if i == 0:
                    maxLeft = nums2[j - 1]
                elif j == 0:
                    maxLeft = nums1[i - 1]
                else:
                    maxLeft = max(nums1[i - 1], nums2[j - 1])

                if i == m:
                    minRight = nums2[j]
                elif j == n:
                    minRight = nums1[i]
                else:
                    minRight = min(nums1[i], nums2[j])

                if (m + n) % 2 == 0:
                    return (maxLeft + minRight) / 2.0
                else:
                    return maxLeft



# 第四题
# Given a string s, find the longest palindromic substring in s.
#  You may assume that the maximum length of s is 1000
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        start=0
        end=0
        length=1
        string_length=len(s)
        dp=[[0]*string_length for i in range(string_length)]
        for i in range(string_length-1,-1,-1):
            for j in range(i,string_length):
                if s[i]==s[j] and ( j-i<3 or dp[i+1][j-1]==1):
                #  print j
                #  if s[i] == s[j] and ( dp[i + 1][j - 1] == 1 or j-i<3):

                    dp[i][j]=1
                    if j-i+1>length:
                        length=j-i+1
                        start=i
                        end=j
        return s[start:end+1]
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        size = len(s)
        if size <= 1 or s == s[::-1]:
            return s
        start, maxlen = 0, 1
        for idx in range(1, size):
            add2 = s[idx - maxlen - 1: idx + 1]
            if idx - maxlen - 1 >= 0 and add2 == add2[::-1]:
                start = idx - maxlen - 1
                maxlen += 2
                continue
            add1 = s[idx - maxlen: idx + 1]
            if add1 == add1[::-1]:
                start = idx - maxlen
                maxlen += 1
        return s[start: (start + maxlen)]
# 第五题：Given a linked list, remove the nth node from the
#  end of list and return its head.
# Given linked list: 1->2->3->4->5, and n = 2.

# After removing the second node from the end,
# the linked list becomes 1->2->3->5.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        length=0
        dummy=ListNode(0)
        dummy.next=head
        first=head
        while first!=None:
            length+=1
            first=first.next
        first=dummy
        length-=n
        while length>0:
            first=first.next
            length-=1
        first.next=first.next.next
        return dummy.next
# 第六题：
# Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.
#
# Your algorithm's runtime complexity must be in the order of O(log n).
#
# If the target is not found in the array, return [-1, -1].
#
# For example,
# Given [5, 7, 7, 8, 8, 10] and target value 8,
# return [3, 4].
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        start=-1
        end=-1
        if  nums==[]:
            return [start,end]
        result_index=self.binarysearch(nums,target)
        if nums[result_index]==target:
            start=result_index
            end=result_index
            while start>0 and nums[start]==nums[start-1]:
                start-=1
            while end<len(nums)-1 and nums[end] == nums[end+1]:
                end+=1
        return [start,end]

    def binarysearch(self,nums,target):
        low,high=0,len(nums)-1
        while low<high:
            middle=(low+high)//2
            if nums[middle]==target:
                return middle
            elif nums[middle]>target:
                high=middle-1
            else:
                low=middle+1
        return low
#********************************************
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int] 1 2 2
        :type target: int
        :rtype: List[int]
        """
        if len(nums) == 0:
            return [-1, -1]
        left = -1
        right = -1
        l = 0
        r = len(nums) - 1

        while l + 1 < r:
            mid = (l + r) / 2
            if target >= nums[mid]:
                l = mid
            else:
                r = mid
        if nums[l] == target:
            right = l
        if nums[r] == target:
            right = r

        l = 0
        r = len(nums) - 1
        while l + 1 < r:  # [5, 7, 7, 8, 8, 10]
            mid = (l + r) / 2
            if target > nums[mid]:
                l = mid
            else:
                r = mid
        if nums[r] == target:
            left = r
        if nums[l] == target:
            left = l

        if left > -1 and right > -1:
            return [left, right]
        elif left > -1:
            return [left, left]
        elif right > -1:
            return [right, right]
        else:
            return [-1, -1]
# 第六题
# 1 is read off as "one 1" or 11.
# 11 is read off as "two 1s" or 21.
# 21 is read off as "one 2, then one 1" or 1211.
# Given an integer n, generate the nth term of the count-and-say sequence.
#
# Note: Each term of the sequence of integers will be represented as a string.


import itertools
class Solution(object):
    def countAndSay(self, n):
        s = '1'
        for _ in range(n - 1):
            s = ''.join(str(len(list(group))) + digit
                        for digit, group in itertools.groupby(s))
        return s

class Solution:
    # @return a string
    def count(self,s):
        t=''; count=0; curr='#'
        for i in s:
            if i!=curr:
                if curr!='#':
                    t+=str(count)+curr
                curr=i
                count=1
            else:
                count+=1
        t+=str(count)+curr
        return t
    def countAndSay(self, n):
        s='1'
        for i in range(2,n+1):
            s=self.count(s)
        return s
# 第七题
# Given an unsorted integer array, find the first missing positive integer.

# For example,
# Given [1,2,0] return 3,
# and [3,4,-1,1] return 2.
#
# Your algorithm should run in O(n) time and uses constant space.
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums==[]:
            return 1
        if 1 not in nums:
            return 1
        nums.sort()
        index=0
        while (nums[index]<0) or ((index<len(nums)-1) and (nums[index]+1==nums[index+1])):
            index+=1
        return nums[index]+1

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n=len(nums)
        f=[False for i in range(0,n+2)]
        for i in range(0,n):
                if ((nums[i]>0)and(nums[i]<=n)): f[nums[i]]=True
        i=1
        while (f[i]):
                i=i+1
        return i
# 第八题
# Given n non-negative integers representing
# an elevation map where the width of each bar is 1,
# compute how much water it is able to trap after raining.
#
# For example,
# Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        mystack=[]
        current=0
        area=0
        while current <len(height):
            while mystack!=[] and height[mystack[-1]]<height[current]:
                top_length=mystack[-1]
                mystack.pop()
                if mystack==[]:
                    break
                distance=current-mystack[-1]-1
                area+=(min(height[current],height[mystack[-1]])-height[top_length])*distance
            mystack.append(current)
            current+=1
        return area
# 第九题
#
# Divide two integers without using multiplication, division and mod operator.
#
# If it is overflow, return MAX_INT.
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        max_int=2147483647
        a= int (float(dividend)/float(divisor))
        return a if a<max_int else max_int

class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        MAX_INT = 2147483647
        quotient = 0
        if((dividend > 0 and divisor) > 0 or (dividend < 0 and divisor < 0)):
            sign = 1
        else:
            sign = -1
        dividend = abs(dividend)
        divisor = abs(divisor)
        while(dividend >= divisor):
            k = 0
            temp = divisor
            while(dividend >= temp):
                dividend -= temp
                quotient += 1 << k
                temp <<= 1
                k += 1
        quotient *= sign
        if quotient > MAX_INT:
            return MAX_INT
        return quotient

# 第十题
# Given a collection of distinct numbers, return all possible permutations.
#
# For example,
# [1,2,3] have the following permutations:
# [
#   [1,2,3],
#   [1,3,2],
#   [2,1,3],
#   [2,3,1],
#   [3,1,2],
#   [3,2,1]
# ]
import itertools
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res=[]
        for i in itertools.permutations(nums):
            res.append(list(i))
        return res

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        for i in range(len(nums)):
            for j in self.permute(nums[:i] + nums[i + 1:]):
                res.append([nums[i]] + j)
        return res
# 第十一题
# Given an array of strings, group anagrams together.
#
# For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
# Return:
# [
#   ["ate", "eat","tea"],
#   ["nat","tan"],
#   ["bat"]
# ]
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dict={}
        for i in strs:
            sort_str=str(sorted(i))
            if sort_str not in dict:
              dict[sort_str]=[]
            dict[sort_str].append(i)
        finalist=[]
        for i in dict.values():
            finalist.append(i)
        return finalist
# *******************
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        store = {}
        res = []
        for string in strs:
            formatted = ''.join(sorted(string))
            if formatted not in store:
                store[formatted] = len(res)
                res.append([string])
            else:
                res[store[formatted]].append(string)

        return res
# 第十二题
# StefanPochmann
# Implement pow(x, n).
#
#
# Example 1:
#
# Input: 2.00000, 10
# Output: 1024.00000
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n - 1)
        return self.myPow(x * x, n / 2)
# ?????
#Find the contiguous subarray within an array (containing at least one number) which has
#  the largest sum.
# For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
# the contiguous subarray [4,-1,2,1] has the largest sum = 6.

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums is None or len(nums) == 0:
            return 0

        localMax, globalMax = nums[0], nums[0]
        n = len(nums)
        for i in range(1, n):
            if localMax < 0:
                localMax = 0
            localMax += nums[i]

            if localMax > globalMax:
                globalMax = localMax
        return globalMax
#?13??
#Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
#
# For example,
# Given the following matrix:
# You should return [1,2,3,6,9,8,7,4,5].

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if  matrix==[]:
            return []
        rowBegin=0
        rowend=len(matrix)-1
        coloumBegin=0
        coloumend=len(matrix[0])-1
        mylist=[]
        while rowBegin<=rowend and coloumBegin<=coloumend:
            for i in range(coloumBegin,coloumend+1):
                mylist.append(matrix[rowBegin][i])
            rowBegin+=1
            for j in range(rowBegin,rowend+1):
                mylist.append(matrix[j][coloumend])
            coloumend-=1
            if rowBegin<=rowend:
                for j in range(coloumend,coloumBegin-1,-1):
                    mylist.append(matrix[rowend][j])
            rowend-=1
            if coloumBegin<=coloumend:
                for i in range(rowend,rowBegin-1,-1):
                    mylist.append(matrix[i][coloumBegin])
            coloumBegin+=1
        return mylist
