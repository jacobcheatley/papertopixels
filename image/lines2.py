import numpy as np
import cv2

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def get_all_lines2(*img_and_labels):
    for img, label in img_and_labels:
        visited = np.zeros(img.shape, np.bool_)

        def exhaust(y_s, x_s):
            segments = []
            branch_points = [(y_s, x_s)]  # To keep track of unfinished branches / joins
            terminals = set()  # To keep track of all branch points / joins

            def neighbors(pix):
                unvisit = []
                term = []
                # Check neighbour directions
                for d in NEIGH:
                    n_y, n_x = pix[0] + d[0], pix[1] + d[1]
                    if img[n_y, n_x]:
                        if not visited[n_y, n_x]:
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
                    visited[curr[0], curr[1]] = True
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
                                visited[curr[0], curr[1]] = True
                                current_segment.append(curr)
                            branch_points.append(curr)
                            terminals.add(curr)
                            end_segment()
                            break

            print('SEGS', '\n'.join(str(l) for l in segments))

        # Search for search start locations
        for y in range(img.shape[0]):
            # TODO: Faster iteration tests
            for x in range(img.shape[1]):
                if visited[y, x]:
                    continue
                elif img[y, x]:
                    exhaust(y, x)
        print('END')


if __name__ == '__main__':
    test = cv2.imread('out/17/lines/r.png', cv2.IMREAD_GRAYSCALE)
    get_all_lines2((test, 'test-red'))