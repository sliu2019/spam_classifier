import re

# count the link in an email
# input: the email title
# output: the number of link
def count_link(title):
	count = 0
	with open(title) as fh:
		for line in fh:
			match = re.search('<a +href="(.+?)" *>', line)
			if (match != None):
				count += 1
	return count

# count the image in an email
# input: the email title
# output: the number of image
def count_image(title):
	count = 0
	with open(title) as html:
		content = html.read()
		matches = re.findall(r'\ssrc="([^"]+)"', content)
		count += len(matches)
	return count


# count the number in an email
# input: the email content
# output: the number of number
def count_number(content):
	count = 0
	with open(title) as fh:
		for line in fh:
			line.split(" ")
			for word in line:
				if is_int(word):
					count += 1
	return count

def is_int(word):
	try: 
		int(word)
		return True
	except ValueError:
		return False


title = "easy_ham/0001.ea7e79d3153e7469e7a9c3e0af6a357e"
print(count_number(title))
print(count_image(title))
