class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(values, method):
    head = None
    for val in reversed(values):
        head = ListNode(val, head)
    
    if method == "iterative":
        prev = None
        curr = head
        while curr is not None:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        new_head = prev

    elif method == "recursive":
        def reverse_recursive(node):
            if node is None or node.next is None:
                return node

            new_head = reverse_recursive(node.next)
            node.next.next = node
            node.next = None
            return new_head

        new_head = reverse_recursive(head)
    else:
        raise ValueError("method must be 'iterative' or 'recursive'")
    
    result = []
    curr = new_head
    
    while curr is not None:
        result.append(curr.val)
        curr = curr.next

    return result
