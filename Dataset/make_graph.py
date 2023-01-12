import re
import subprocess
import javalang
import json


MODIFIERS = ['abstract', 'default', 'final', 'native', 'private',
                  'protected', 'public', 'static', 'strictfp', 
                  'transient', 'volatile'] 


class Node:
    def __init__(self):
        self.ori_id = None
        self.idx = None
        self.type = None
        self.label = None
        self.typeLabel = None
        self.pos = None
        self.length = None
        self.children = []
        self.father = None
    def __str__(self):
        return 'ori_id:%s idx:%s type:%s label:%s typeLabel:%s pos:%s length:%s'%(self.ori_id, self.idx, self.type, self.label, self.typeLabel, self.pos, self.length)
    def print_tree(self, depth):
        strr = '\t' * depth + str(self) + '\n'
        for child in self.children:
            strr += child.print_tree(depth + 1)
        return strr
    def get_all_nodes(self):
        nodes = []
        nodes.append(self)
        for node in self.children:
            nodes += node.get_all_nodes()
        return nodes

class ActionNode:
    def __init__(self, typ, idx, name=None):
        self.typ = typ
        self.idx = idx
        self.name = name
    
    def __eq__(self, node):
        return self.typ == node.typ and self.idx == node.idx and self.name == node.name

    def __str__(self):
        return f'typ = {self.typ}, idx = {self.idx}, name = {self.name}'


def extract_segments(chunk):

    diff_lines = chunk.split('\n')
 
    deleted_code = ""
    added_code = ""

    for line in diff_lines:

        if not line:
            continue

        if line[0] == '-':
            line = line[1:].strip()
            deleted_code += line

        elif line[0] == '+':
            line = line[1:].strip()
            added_code += line
        
    return deleted_code, added_code


def process_bracket(tokens):
    if tokens[0] == '}':
        tokens.pop(0)
    stack = []
    for token in tokens:
        if token == '{':
            stack.append('{')
        elif token == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                stack.append('}')
    l_num = stack.count('{')
    r_num = stack.count('}')
    tokens = ['{'] * r_num + tokens + ['}'] * l_num
    return tokens


def get_ast(text, file_name):

    if len(text) == 0:
        print("No code.")
        return None, -1

    try:
        tokens_ori = list(javalang.tokenizer.tokenize(text))
    except:
        print("Tokenization error.")
        return None, -1
        
    codes_ori = [x.value for x in tokens_ori]

    if len(codes_ori) == 0: 
        print("No Tokens") 
        return None, -1
    
    if 'implement' in codes_ori:
        codes_ori.remove('implement')
    if codes_ori[-1] == 'implements':
        codes_ori.remove('implements')
    if len(codes_ori) == 0:
        print("No Tokens after filtering") 
        return None, -1
    
    
    if len(codes_ori) >= 4 and 'class' in codes_ori and codes_ori[-2] == '<' and codes_ori[-1] != '>':
        codes_ori += '>' 
    
    codes_ori = process_bracket(codes_ori)
    
    if len(codes_ori) == 0:
        print("No Tokens after brackets") 
        return None, -1
    
    ori_start_token = ' '.join(codes_ori)
    
    if codes_ori[0] == 'import':
        pass
    elif codes_ori[0] == 'package':
        pass
    elif codes_ori[0] == '@':
        if 'class' in codes_ori:  # definition of class
            pass
        else:  # definition of method
            codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
            # gumtree can only parse class, so a padding class needs to be inserted
    elif codes_ori[0] in MODIFIERS:
        
        if 'class' in codes_ori:  # definition of class
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                raise
            else:
                codes_ori +=  ['{', '}'] 
        elif '(' in codes_ori and ')' in codes_ori and ('=' not in codes_ori or ('='  in codes_ori and codes_ori.index('(') < codes_ori.index('=') and codes_ori.index(')') < codes_ori.index('='))):  # definition of method
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                raise
            elif codes_ori[-1] != ';':
                codes_ori +=  ['{', '}'] 
                
            codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
        else:  # definition of field
            codes_ori = ['class', 'pad_pad_class', '{', '{'] + codes_ori + ['}', '}']
    elif codes_ori[0] == '{':
        codes_ori = ['class', 'pad_pad_class', '{'] + codes_ori + ['}']
    else:
        if codes_ori[0] == 'if':
            if codes_ori[-1] == '}':
                pass
            elif codes_ori[-1] == '{':
                
                raise
            elif codes_ori[-1] == ')':
                codes_ori +=  ['{', '}']
        codes_ori = ['class', 'pad_pad_class', '{', '{'] + codes_ori + ['}', '}']

  
    text = ' '.join(codes_ori)
    start_code_pos = text.index(ori_start_token)
    assert start_code_pos != -1

    open('%s.java'%file_name, 'w+').write(text)
    
    root = get_ast_root(file_name)

    return root

def get_ast_root(file_name):

    out = subprocess.Popen('../lib/gumtree/gumtree/bin/gumtree parse %s.java'%(file_name), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,_ = out.communicate()
    try:
        ast = json.loads(stdout.decode('utf-8'))
    except:
        print("error: gum tree parse.")
        return None
    json.dump(ast, open('%s.ast'%(file_name), 'w'), indent=1)
    root = Node()
    root.label = 'root'
    root.pos = -1
    all_nodes = []
    all_nodes.append(root)
    all_nodes += process_ast(ast['root'])
    all_nodes += ['^']

    all_nodes_new = []
    root = all_nodes[0]
    root.idx = 0
    all_nodes_new.append(root)
    cur_node = root
    idx = 1
    for node in all_nodes[1:]:
        if node == '^':
            cur_node = cur_node.father
        else:
            node.idx = idx
            node.father = cur_node
            all_nodes_new.append(node)
            cur_node.children.append(node)
            cur_node = node
            idx += 1
    return root


def process_ast(ast):
    nodes = []
    
    node = Node()
    if 'label' in ast:
        node.label = ast['label']
    else:
        node.label = None
    
    node.ori_id = int(ast['id'])
    node.type = ast['type']
    node.typeLabel = ast['typeLabel']
    node.pos = int(ast['pos'])
    node.length = ast['length']

    if node.typeLabel == 'NullLiteral':
        assert node.label == None
        node.label = 'null'
    if node.typeLabel == 'ThisExpression':
        assert node.label == None
        node.label = 'this'
    nodes.append(node)

    for child in ast["children"]:
        nodes += process_ast(child)
        nodes.append('^')
    return nodes



def get_typ_idx(strr):
    if ':' in strr:
        typ, name_idx = strr.split(':')
        typ = typ.strip()
        name_idx = name_idx.strip()
        name = name_idx[:name_idx.index('(')]
        idx = name_idx[name_idx.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        return ActionNode(typ, int(idx), name)

    else:    
        typ = strr[:strr.index('(')]
        idx = strr[strr.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        if typ == 'NullLiteral':
            return ActionNode(typ, int(idx), 'null')
        if typ == 'ThisExpression':
            return ActionNode(typ, int(idx), 'this')
        return ActionNode(typ, int(idx))

def get_ast_action(file_name1, file_name2, root1, root2):

    out = subprocess.Popen('../lib/gumtree/gumtree/bin/gumtree diff %s.java %s.java'%(file_name1, file_name2), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    stdout = stdout.decode('utf-8')
    raw_actions = [x.strip() for x in stdout.splitlines() if x.strip()]

    all_match = []
    all_delete = []
    all_update = []
    all_move = []
    all_add = []
    for i, raw_action in enumerate(raw_actions):
        if raw_action.startswith('Match'):
            raw_node_old, raw_node_new = raw_action.lstrip('Match').split(' to ')
            raw_node_old = raw_node_old.strip()
            raw_node_new = raw_node_new.strip()
            node_old = get_typ_idx(raw_node_old)
            node_new = get_typ_idx(raw_node_new)
            all_match.append((node_old, node_new))
        elif raw_action.startswith('Delete'):
            raw_node_old = raw_action.lstrip('Delete')
            raw_node_old = raw_node_old.strip()
            node_old = get_typ_idx(raw_node_old)
            all_delete.append(node_old)
        elif raw_action.startswith('Update'):
            raw_node_old, new_name = raw_action.lstrip('Update').split(' to ')
            raw_node_old = raw_node_old.strip()
            new_name = new_name.strip()
            node_old = get_typ_idx(raw_node_old)
            all_update.append((node_old, new_name))
        elif raw_action.startswith('Move'):
            raw_node_old, tem = raw_action.lstrip('Move').split(' into ')
            raw_node_new, new_pos = tem.split(' at ')
            raw_node_old = raw_node_old.strip()
            raw_node_new = raw_node_new.strip()
            new_pos = new_pos.strip()
            node_old = get_typ_idx(raw_node_old)
            node_new = get_typ_idx(raw_node_new)
            new_pos = int(new_pos)
            all_move.append((node_old, node_new, new_pos))
        elif raw_action.startswith('Insert'):
            raw_node_new, tem = raw_action.lstrip('Insert').split(' into ')
            raw_node_par, pos = tem.split(' at ')
            raw_node_new = raw_node_new.strip()
            raw_node_par = raw_node_par.strip()
            pos = pos.strip()
            node_new = get_typ_idx(raw_node_new)
            node_par = get_typ_idx(raw_node_par)
            pos = int(pos)
            all_add.append((node_new, node_par, pos))

    all_nodes1 = root1.get_all_nodes()
    all_nodes2 = root2.get_all_nodes()
    map1 = {}
    map2 = {}  

    for i, node in enumerate(all_nodes1):
        map1[node.ori_id] = i 
        assert i == node.idx
    
    for i, node in enumerate(all_nodes2):
        map2[node.ori_id] = i 
        assert i == node.idx
    
    document_move = [False] * len(all_move)
    document_update = [False] * len(all_update)
    all_match_new = []

    for i in range(len(all_match)):
        cur_match = all_match[i]
        move_flag = False
        update_flag = False
        
        for j in range(len(all_update)):
            cur_update = all_update[j]
            if cur_update[0] == cur_match[0]:
                value1 = cur_update[1]
                value2 = cur_match[1].name
                assert value1 == value2
                update_flag = True
                document_update[j] = True
                break

        for j in range(len(all_move)):
            cur_move = all_move[j]
            if cur_move[0] == cur_match[0]:
                parent_id = cur_move[1].idx
                child_id = cur_match[1].idx
                children_id = [x.ori_id for x in all_nodes2[map2[parent_id]].children]
                assert child_id in children_id
                move_flag = True
                document_move[j] = True
                if update_flag == False:
                    assert cur_match[0].typ == cur_match[1].typ
                    assert cur_match[0].name == cur_match[1].name
                break

        if move_flag == False and update_flag == False:
            all_match_new.append(('match', cur_match[0], cur_match[1]))
        elif move_flag == False and update_flag == True:
            all_match_new.append(('update', cur_match[0], cur_match[1]))
        elif move_flag == True and update_flag == False:
            all_match_new.append(('move', cur_match[0], cur_match[1]))
        elif move_flag == True and update_flag == True:
            all_match_new.append(('update', cur_match[0], cur_match[1]))
    
    assert sum(document_move) == len(all_move)
    assert sum(document_update) == len(all_update)
    
    for i in range(len(all_add)):
        cur_add = all_add[i]
        child_id = cur_add[0].idx
        parent_id = cur_add[1].idx
        children_id = [x.ori_id for x in all_nodes2[map2[parent_id]].children]
        assert child_id in children_id

    return all_match_new, all_delete, all_add


def traverse_graph(root: Node, graph, idx, pidx, reverse_map, reverse_map_pos, nodeType):

    idx0 = idx[0]

    # nodeType can be one of {addedAST, deletedAST, action}

    graph[str(idx0)] = {
        'ori_id': root.ori_id,
        'idx': str(idx0),
        'nodeType': nodeType,
        'type': root.type,
        'typeLabel': root.typeLabel,
        'label': root.label,
        'pos': root.pos,
        'links': [pidx] if pidx is not None else [],
        'link_type': ['normal'] if pidx is not None else [],
    }

    reverse_map[f'{nodeType}_{root.ori_id}'] = idx0
    reverse_map_pos[f'{nodeType}_{root.pos}'] = idx0

    for child in root.children:

        graph[str(idx0)]['links'].append(idx[0]+1)
        graph[str(idx0)]['link_type'].append('normal')
        idx[0] += 1
        traverse_graph(child, graph, idx, idx0, reverse_map, reverse_map_pos, nodeType)


if __name__=='__main__':

    diff_text = open('diff.txt').read()

    # divide them into chunks
    chunks = re.split("@@[\s][-][0-9]+[\,][0-9]+[\s][+][0-9]+[\,][0-9]+[\s]@@", diff_text)

    # first one is empty
    chunks = chunks[1:]

    for chunk in chunks:
        deleted_text, added_text = extract_segments(chunk)

        deleted_text, added_text = deleted_text+';', added_text+';'

        # deleted_text, added_text = "", ""
        # print(deleted_text, added_text)
        del_root: Node = get_ast(deleted_text, 'deleted')
        add_root = get_ast(added_text, 'added')

        # print(del_root, add_root)
        # all_match_new -> match, move, update
        try:
            all_match_new, all_delete, all_add = get_ast_action('deleted', 'added', del_root, add_root)
        except Exception as e:
            print(e)
            continue

        graph = {}
        reverse_map = {}
        reverse_map_pos = {}

        idx = [0]
        traverse_graph(del_root, graph, idx, None, reverse_map, reverse_map_pos, 'deletedAst')
        idx[0] += 1
        traverse_graph(add_root, graph, idx, None, reverse_map, reverse_map_pos, 'addedAst')
        idx[0] += 1

        # Add all action nodes -----

        for node in all_match_new:

            link1 = reverse_map[f'deletedAst_{node[1].idx}']
            link2 = reverse_map[f'addedAst_{node[2].idx}']

            action_node = {
                'ori_id': None,
                'idx': str(idx[0]),
                'nodeType': 'action',
                'type': None,
                'typeLabel': None,
                'label': node[0], # match / move / update
                'links': [link1, link2],
                'link_type': ['action', 'action']
            }

            graph[str(link1)]['links'].append(idx[0])
            graph[str(link1)]['link_type'].append('action')
            graph[str(link2)]['links'].append(idx[0])
            graph[str(link2)]['link_type'].append('action')
            graph[str(idx[0])] = action_node

            idx[0] += 1

            # print(f'{node[0]}; {node[1]}; {node[2]}')
    
        
        for node in all_delete:
            link1 = reverse_map[f'deletedAst_{node.idx}']

            action_node = {
                'ori_id': None,
                'idx': str(idx[0]),
                'nodeType': 'action',
                'type': None,
                'typeLabel': None,
                'label': 'delete',
                'links': [link1],
                'link_type': ['action']
            }

            graph[str(link1)]['links'].append(idx[0])
            graph[str(link1)]['link_type'].append('action')
            graph[str(idx[0])] = action_node

            idx[0] += 1

        for node in all_add:
            link1 = reverse_map[f'addedAst_{node[0].idx}']

            action_node = {
                'ori_id': None,
                'idx': str(idx[0]),
                'nodeType': 'action',
                'type': None,
                'typeLabel': None,
                'label': 'add',
                'links': [link1],
                'link_type': ['action']
            }

            graph[str(link1)]['links'].append(idx[0])
            graph[str(link1)]['link_type'].append('action')
            graph[str(idx[0])] = action_node

            idx[0] += 1


        # Sequential type edges ------------------

        deleted_pos = [graph[node]['pos'] for node in graph if graph[node]['nodeType']=='deletedAst' and (graph[node]['label'] is not None or graph[node]['typeLabel']=='ReturnStatement')]
        added_pos = [graph[node]['pos'] for node in graph if graph[node]['nodeType']=='addedAst' and (graph[node]['label'] is not None or graph[node]['typeLabel']=='ReturnStatement')]

        deleted_pos = sorted(deleted_pos)
        added_pos = sorted(added_pos)

        # for node in graph.keys():
        #     if graph[node]['nodeType'] == 'addedAst':
        #         added_pos.append(graph[node]['pos'])
        #     elif graph[node]['nodeType'] == 'deletedAst':
        #         deleted_pos.append(graph[node]['pos'])

        for i in range(len(deleted_pos)-1):

            pos = deleted_pos[i]
            next_pos = deleted_pos[i+1]

            idx1 = reverse_map_pos[f'deletedAst_{pos}']
            idx2 = reverse_map_pos[f'deletedAst_{next_pos}']

            graph[str(idx1)]['links'].append(idx2)
            graph[str(idx2)]['links'].append(idx1)
            graph[str(idx1)]['link_type'].append('seq')
            graph[str(idx2)]['link_type'].append('seq')

        for i in range(len(added_pos)-1):

            pos = added_pos[i]
            next_pos = added_pos[i+1]

            idx1 = reverse_map_pos[f'addedAst_{pos}']
            idx2 = reverse_map_pos[f'addedAst_{next_pos}']

            graph[str(idx1)]['links'].append(idx2)
            graph[str(idx2)]['links'].append(idx1)
            graph[str(idx1)]['link_type'].append('seq')
            graph[str(idx2)]['link_type'].append('seq')

        with open('graph.json', 'w+') as f:
            json.dump(graph, f)

        break





