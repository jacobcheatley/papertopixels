import numpy as np
import cv2
from collections import defaultdict
from image.cycles import simple_cycles

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_COMP = 10


def end_complement(path):
    return [(seg[0], -1 if seg[1] == 0 else 0) for seg in path]


def join_segments(segments):
    """Figure out how all the segments should join, then return all simplified lines"""

    print('COMPONENT')
    for seg in segments:
        print('-seg', seg[0], seg[-1], len(seg))

    '''
    # Many cases will just be 1 or 2 segments, easier and faster to manually do
    if len(segments) == 1:  # straight line or single loop
        seg = segments[0]
        if seg[0] == seg[-1]:  # loop
            return [(seg[:-1], True)]
        else:
            return [(seg, False)]
    elif len(segments) == 2:  # straight line, or loop with line picked up first
        seg1, seg2 = segments[0], segments[1]
        if seg2[0] == seg2[-1]:
            return [(seg2[:-1], True), (seg1, False)]
        else:
            # Try to join up
            if seg1[0] == seg2[0]:
                return [(reversed(seg1[1:] + seg2), False)]
            else:
                print('Uuuhh - THIS SHOULDN\'T HAPPEN')
                return []
    '''

    final = []

    # Generating graph of linked sections
    graph = defaultdict(list)  # (index, end (0/-1)): [(index, end (0/-1))]
    removed = set()
    for i1, seg1 in enumerate(segments):
        for i2, seg2 in enumerate(segments):
            if seg2[0] == seg2[-1]:  # Remove self joining loops
                final.append(([seg1[:-1]], True))
                removed.add(i1)
                continue
            # Isolated loop ignore
            if i2 in removed: continue

            # Test for connectivity
            for t1, t2 in [(0, 0), (-1, -1), (0, -1), (-1, 0)]:
                if (i1, t1) == (i2, t2): continue
                if seg1[t1] == seg2[t2]:
                    graph[(i1, t1)].append((i2, t2))
                    # graph[i1].append(i2)

    print('graph', graph)

    # Find all paths through the graph
    # Because of the way graph is structured, this is not just cycles
    # Reverse order of length
    paths = sorted(list(simple_cycles(graph)) + [[(single, None)] for single in range(len(segments))],
                   key=lambda p: sum(len(segments[s[0]]) for s in p) - len(p))
    print('paths', paths)
    cycles = [p for p in paths if end_complement(p) in paths]
    print('cycles', cycles)

    chosen_cycles = []
    while cycles:
        # Grab the largest cycle, remove cycles containing those segments
        largest = cycles.pop()
        chosen_cycles.append(largest)
        rmv = [c[0] for c in largest]
        # NONE of the segments can be shared
        cycles = [p for p in cycles if not any(s in rmv for s, _ in p)]
        paths = [p for p in paths if not any(s in rmv for s, _ in p)]

    chosen_paths = []
    while paths:
        # Same as cycles, but don't need to care about cycles list as well
        largest = paths.pop()
        chosen_paths.append(largest)
        rmv = [c[0] for c in largest]
        paths = [p for p in paths if not any(s in rmv for s, _ in p)]

    print('FINAL cycles', chosen_cycles)
    print('FINAL paths', chosen_paths)

    return [([(0, 0), (1, 1)], True)]


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

            return segments if comp_pixels[0] > MIN_COMP else None

        # Search for search start locations
        for y in range(img.shape[0]):
            # TODO: Faster iteration tests
            for x in range(img.shape[1]):
                if (y, x) in visited:
                    continue
                elif img[y, x]:
                    segments = exhaust(y, x)
                    # Segments is none if the total shape is too small
                    if segments is not None:
                        for points, closed in join_segments(segments):
                            print('line', points[0], points[1], len(points), closed)
                            yield {
                                'color': label,
                                'closed': closed,
                                'points': [{'x': x, 'y': y} for x, y in points],
                            }


if __name__ == '__main__':
    test = cv2.imread('out/17/lines/r.png', cv2.IMREAD_GRAYSCALE)
    list(get_all_lines2((test, 'test-red')))
