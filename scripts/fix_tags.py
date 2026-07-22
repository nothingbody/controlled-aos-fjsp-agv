"""Replace all remaining \\tag{X} with \\qquad \\mathrm{(X)} in the paper."""
import re

with open('SCI_Paper/GRL_EA_paper.md', 'r', encoding='utf-8') as f:
    content = f.read()

count_before = len(re.findall(r'\\tag\{', content))
print(f'Before: {count_before} \\tag occurrences')

# Replace \tag{X} -> \qquad \mathrm{(X)}
def replacer(m):
    val = m.group(1)
    return r'  \qquad \mathrm{(' + val + ')}'

content = re.sub(r'\\tag\{([^}]+)\}', replacer, content)

count_after = len(re.findall(r'\\tag\{', content))
print(f'After: {count_after} \\tag occurrences')

with open('SCI_Paper/GRL_EA_paper.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
