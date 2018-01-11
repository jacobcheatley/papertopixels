import numpy as np
import cv2
from collections import defaultdict
from image.cycles import simple_cycles

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_COMP = 10


def join_segments(segments, terminals):
    """Figure out how all the segments should join, then return all simplified lines"""
    print('COMPONENT')
    for index, seg in enumerate(segments):
        print(f'-seg{index}', seg[0], seg[-1], len(seg))

    print(terminals)
    removed = set()  # for single loops and removed loops
    graph = defaultdict(list)  # (index, term?): [(index, term?)]

    def all_paths(v):
        """Generate the maximal cycle-free paths in graph starting at v.
        graph must be a mapping from vertices to collections of
        neighbouring vertices."""
        path = [v]  # path traversed so far
        seen = {v}  # set of vertices in path

        def search():
            dead_end = True
            for neighbour in graph[path[-1]]:
                if neighbour not in seen:
                    dead_end = False
                    seen.add(neighbour)
                    path.append(neighbour)
                    yield from search()
                    path.pop()
                    seen.remove(neighbour)
            if dead_end and len(path) % 2 == 1:
                yield list(path)

        yield from search()

    def path_len(path):
        total = 0
        for node, term in path:
            if not term:
                total += len(segments[node])
            else:
                total -= 1
        return total

    # Generate the graph
    for si, seg in enumerate(segments):
        if seg[0] == seg[-1]:
            removed.add(si)
            continue
        for ti, term in enumerate(terminals):
            if seg[0] == term or seg[-1] == term:
                graph[(ti, True)].append((si, False))
                graph[(si, False)].append((ti, True))

    # Get out the largest cycles
    print('GRAPH', graph)

    cycles = sorted([c for c in simple_cycles(graph) if len(c) > 2], key=path_len)
    chosen_cycles = []
    while cycles:
        # Grab largest, remove things that share segments
        largest = cycles.pop()
        chosen_cycles.append(largest)
        rmv = [node for node in largest if not node[1]]  # Remove with matching segment
        for seg in rmv:
            # Doing removal
            removed.add(seg[0])
            del graph[seg]
        for key in graph:
            if key[1]:  # Only bother looking at terminals
                neighs = graph[key]
                graph[key] = [n for n in neighs if n not in rmv]
        cycles = [c for c in cycles if not any(n in rmv for n in c)]

    print('CYCLES', chosen_cycles)
    print('GRAPH', graph)
    print('REMOVED', removed)

    # Grab all possible paths
    found_paths = []
    for si in range(len(segments)):
        if si in removed: continue
        found_paths.extend(all_paths((si, False)))
        found_paths.append([(si, False)])

    found_paths.sort(key=path_len)

    print('PATHS', found_paths)
    print('LENGTHS', [path_len(path) for path in found_paths])

    # Get out the largest linear paths
    chosen_paths = []
    while found_paths:
        # Grab largest, remove things that share segments
        largest = found_paths.pop()
        chosen_paths.append(largest)
        rmv = [node for node in largest if not node[1]]  # Remove with matching segment
        found_paths = [p for p in found_paths if not any(n in rmv for n in p)]

    print('FINAL CYCLES', chosen_cycles)
    print('FINAL PATHS', chosen_paths)

    return []


def get_all_lines2(*img_and_labels):
    """From each color channel, yield out all the labelled lines"""
    for img, label in img_and_labels:
        print(label)
        visited = set()

        def exhaust(y_s, x_s):
            """Exhaust an entire component to extract partial segments"""
            segments = []
            branch_points = [(y_s, x_s)]  # To keep track of unfinished branches / joins
            terminals = set()  # To keep track of all branch points / joins
            comp_pixels = [0]  # To keep track of total component size. Have to keep in list to assign, gross

            def neighbors(pix):
                unvisit = []
                term = []
                # Check neighbour directions
                for d in NEIGH:
                    n_y, n_x = pix[0] + d[0], pix[1] + d[1]
                    if img[n_y, n_x]:
                        if (n_y, n_x) not in visited:
                            unvisit.append((n_y, n_x))
                        elif (n_y, n_x) in terminals:
                            term.append((n_y, n_x))
                            # Give back all unvisited neighbours AND terminals
                return unvisit, term

            def count_total_neighbors(pix):
                result = 0
                for d in NEIGH:
                    n_y, n_x = pix[0] + d[0], pix[1] + d[1]
                    if img[n_y, n_x]:
                        result += 1
                return result

            def get_start_point():
                while branch_points:
                    test = branch_points.pop()
                    neigh, _ = neighbors(test)
                    if neigh:
                        return test
                return None

            def visit(pix):
                if pix not in visited:
                    comp_pixels[0] += 1
                    visited.add(pix)

            # Keep going back while there are unfinished branches in this component
            while True:
                curr = get_start_point()
                if curr is None:
                    break

                current_segment = []

                def end_segment():
                    segments.append(current_segment[:])
                    current_segment.clear()

                while True:
                    visit(curr)
                    current_segment.append(curr)
                    neigh, terms = neighbors(curr)

                    if terms and len(current_segment) > 2:
                        # Deal with ending
                        current_segment.append(terms[0])
                        end_segment()
                        break
                    else:
                        adj = len(neigh)
                        if adj == 0:
                            # End
                            segments.append(current_segment[:])
                            current_segment.clear()
                            break
                        elif adj == 1:
                            # Single continuation
                            curr = neigh[0]
                        elif adj == 2 and len(current_segment) == 1:
                            # Beginning of two way
                            branch_points.append(curr)
                            terminals.add(curr)
                            curr = neigh[0]
                        else:
                            # Must be a branch point - check that a neighbour doesn't have more places to go
                            # As THAT would be the true branch point, so continue to that
                            counts = [count_total_neighbors(pix) for pix in neigh]
                            highest = max(counts)
                            if highest > adj:
                                # Use the NEXT point as branch point
                                curr = neigh[counts.index(highest)]
                                visit(curr)
                                current_segment.append(curr)
                            branch_points.append(curr)
                            terminals.add(curr)
                            end_segment()
                            break

            return (segments, terminals) if comp_pixels[0] > MIN_COMP else (None, None)

        # Search for search start locations
        for y in range(img.shape[0]):
            # TODO: Faster iteration tests
            for x in range(img.shape[1]):
                if (y, x) in visited:
                    continue
                elif img[y, x]:
                    segments, terminals = exhaust(y, x)
                    # Segments is none if the total shape is too small
                    if segments is not None:
                        join_segments(segments, terminals)

        yield None  # TODO: Remove this


if __name__ == '__main__':
    test = cv2.imread('data/mawp.png', cv2.IMREAD_GRAYSCALE)
    list(get_all_lines2((test, 'test')))
