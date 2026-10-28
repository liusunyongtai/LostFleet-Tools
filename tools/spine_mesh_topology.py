"""Editor-oriented mesh checks beyond runtime finite-vertex validation.

A regular image mesh must triangulate one UV disk, with one perimeter. Spine
nonessential edges use *coordinate* indices (2 * vertex index).
"""
from collections import Counter,defaultdict
import math


def perimeter_edges(hull):
    if hull<3:raise ValueError('A mesh perimeter requires at least three vertices')
    return [0,(hull-1)*2]+[v for i in range(hull-1) for v in (i*2,(i+1)*2)]


def editor_mesh_check(mesh):
    errors=[];uv=mesh.get('uvs',[]);triangles=mesh.get('triangles',[])
    if len(uv)%2 or len(uv)<6:return ['invalid UV coordinate count']
    points=list(zip(uv[::2],uv[1::2]));n=len(points)
    if not all(math.isfinite(x) and math.isfinite(y) for x,y in points):errors.append('non-finite UV')
    if len(set(points))!=n:errors.append('duplicate UV vertices')
    if len(triangles)%3:return errors+['invalid triangle count']
    edges=Counter();neighbors=defaultdict(set);area=0.;used=set()
    for i in range(0,len(triangles),3):
        a,b,c=triangles[i:i+3]
        if any(not isinstance(v,int) or v<0 or v>=n for v in (a,b,c)):
            errors.append('triangle vertex outside mesh');continue
        p,q,r=points[a],points[b],points[c];signed=(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);area+=signed/2
        if signed<=1e-12:errors.append('degenerate or reversed UV triangle')
        used.update((a,b,c))
        for x,y in ((a,b),(b,c),(c,a)):
            edges[tuple(sorted((x,y)))]+=1;neighbors[x].add(y);neighbors[y].add(x)
    if used!=set(range(n)):errors.append('unused UV vertices')
    seen=set();stack=[0]
    while stack:
        v=stack.pop()
        if v not in seen:seen.add(v);stack.extend(neighbors[v]-seen)
    if len(seen)!=n:errors.append('disconnected mesh islands')
    if any(v>2 for v in edges.values()):errors.append('non-manifold edge')
    if n-len(edges)+len(triangles)//3!=1:errors.append('mesh is not a triangulated disk')
    hull=mesh.get('hull',0)
    if not 3<=hull<=n:errors.append('missing or invalid hull')
    else:
        expected={tuple(sorted((i,(i+1)%hull))) for i in range(hull)};boundary={e for e,c in edges.items() if c==1}
        if boundary!=expected:errors.append('triangle boundary does not match ordered hull')
        metadata=mesh.get('edges')
        if metadata is None:errors.append('missing editor edges')
        elif len(metadata)%2 or any(not isinstance(v,int) or v<0 or v>=n*2 or v%2 for v in metadata):errors.append('invalid editor edge coordinates')
        else:
            fixed={tuple(sorted((a//2,b//2))) for a,b in zip(metadata[::2],metadata[1::2])}
            if not expected<=fixed:errors.append('editor edges omit hull boundary')
        hullarea=sum(points[i][0]*points[(i+1)%hull][1]-points[(i+1)%hull][0]*points[i][1] for i in range(hull))/2
        if not math.isclose(area,hullarea,abs_tol=1e-7):errors.append('UV triangles do not cover hull exactly once')
    return sorted(set(errors))
