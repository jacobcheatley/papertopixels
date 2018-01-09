import numpy as np
import cv2

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_COMP = 10


def join_segments(segments):
    """Figure out how all the segments should join, then return all simplified lines"""
    # First, join up everything with EXACTLY ONE partner into new lines
    # While there are still lines that can be joined, test combinations for loop creation, then longest line
    # When there are no more matching ends, return all joined lines
    print('COMPONENT')
    finished = []
    for seg in segments:
        print('SEG-INFO', seg[0], seg[-1], len(seg))
    return finished


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
                            yield {
                                'color': label,
                                'closed': closed,
                                'points': [{'x': x, 'y': y} for x, y in points.tolist()],
                            }


if __name__ == '__main__':
    test = cv2.imread('out/17/lines/g.png', cv2.IMREAD_GRAYSCALE)
    list(get_all_lines2((test, 'test-red')))
