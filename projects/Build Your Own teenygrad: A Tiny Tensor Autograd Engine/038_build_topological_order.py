def build_topological_order(tensor):
    visited = set()
    order = []

    def dfs(node):
        visited.add(id(node))
        if node._ctx is not None:
            for p in node._ctx.parents:
                if id(p) not in visited:
                    dfs(p)
        order.append(node)

    dfs(tensor)
    return order
