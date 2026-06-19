from bs4 import BeautifulSoup

with open('docs/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
scripts = soup.find_all('script')
# The last script should be the inline one, find its position
inline_script = None
for s in scripts:
    if s.get('src', '') == '':
        inline_script = s
        break

if inline_script:
    body = soup.find('body')
    body_children = list(body.children)
    last_child = None
    for child in reversed(body_children):
        if str(child).strip():
            last_child = child
            break
    
    # Check if inline script is last meaningful child
    print('Inline script is last meaningful child:', inline_script == last_child)
    print('Has #root before script:', 'id="root"' in str(soup) and str(inline_script) in str(soup))
    
    # Verify script position relative to content
    html_str = str(soup)
    root_pos = html_str.find('id="root"')
    loading_pos = html_str.find('id="loading"')
    content_pos = html_str.find('id="content"')
    script_pos = html_str.find('function render()')
    print('root position:', root_pos)
    print('loading position:', loading_pos)
    print('content position:', content_pos)
    print('script position:', script_pos)
    print('Script AFTER content?', script_pos > content_pos if script_pos > 0 and content_pos > 0 else 'not found')
    print('Script AFTER loading?', script_pos > loading_pos if script_pos > 0 and loading_pos > 0 else 'not found')
else:
    print('ERROR: no inline script found')

print()
print('Total script tags:', len(scripts))
for i, s in enumerate(scripts):
    print(f'  [{i}] src={s.get("src", "")}')
