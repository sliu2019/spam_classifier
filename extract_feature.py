import re

# Count the link in an email
# Input: title of an txt file
# Output: the number of link
def count_link(content):
    regex = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    matches = re.findall(regex, content)
    return len(matches)


# Count the image in an email
def count_image(content):
    regex = r'\ssrc="([^"]+)"'
    matches = re.findall(regex, content)
    return len(matches)

# Count the number in an email
def count_number(content):
    matches = [s for s in content.split() if s.isnumeric()]
    return len(matches)


# Count the number of phone number
def count_phone(content):
    regex = r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$'
    matches = re.findall(regex, content)
    return len(matches)


# Test case
# title = "easy_ham/0009.435ae292d75abb1ca492dcc2d5cf1570"
# print(count_link(title))
# print(count_image(title))
# print(count_number(title))
# print(count_phone(title))

