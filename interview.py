# python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
import math
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        curList = []
        for head in lists:
            curList.append(head)
        min_idx, min_node = self.find_min(curList)
        pHead = ListNode(min_node.val)
        curList[min_idx] = curList[min_idx].next
        while self.notNone(curList):
            min_idx, min_node = self.find_min(curList)
            pHead

    def notNone(self, curList):
        for node in curList:
            if node:
                return True
        return False

    def find_min(self, curList):
        min_idx = -1
        min_val = math.inf
        min_node = None
        for idx, node in enumerate(curList):
            if node.val < min_val:
                min_val = node.val
                min_node = node
                min_idx = idx
        return min_idx, min_node
