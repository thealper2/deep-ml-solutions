def valid_palindrome(s):
    def is_palindrome_range(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            
            left += 1
            right -= 1
        
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return is_palindrome_range(left + 1, right) or is_palindrome_range(left, right - 1)
        
        left += 1
        right -= 1
    
    return True
