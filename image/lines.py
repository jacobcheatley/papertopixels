import numpy as np
import cv2
from collections import defaultdict
from image.cycles import simple_cycles

# CONSTANTS
EPSILON = 1
NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_COMP = 10
DEBUG_PRINT = False
# COLOR
BLACK_ABS_V = 65
BLACK_TEST_V = 120
BLACK_TEST_S = 30
EXTENT = 1


def all_paths(v, graph):
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


class LineFinder:
    def __init__(self, img, thinned):
        self.img = img
        self.hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Process HSV image to blacken already blackish pixels -> VERY slow, need a better way
        # for y in range(self.hsv_img.shape[0]):
        #     for x in range(self.hsv_img.shape[1]):
        #         v = self.hsv_img[y, x, 2]
        #         if v < BLACK_ABS_V:
        #             self.hsv_img[y, x, 2] = 0
        #         else:
        #             self.hsv_img[y, x, 2] += 10
        self.thinned = thinned

    def _color(self, segment):
        """Return b g r or k for the segment, by averaging color along it"""
        # TODO: This will have to be adapative somehow as blacks get miscategorised
        pixels = np.array([self.hsv_img[y-EXTENT:y+EXTENT+1, x-EXTENT:x+EXTENT+1] for y, x in segment])
        pixels = pixels.reshape(-1, pixels.shape[-1])  # Remove unneeded extra dimensions
        v = [p[2] for p in pixels]

        v_avg = np.average(v)
        if v_avg < BLACK_ABS_V:
            return 'k'

        if v_avg < BLACK_TEST_V:
            s = [p[1] for p in pixels]
            s_avg = np.average(s)
            if s_avg < BLACK_TEST_S:
                return 'k'

        h = [p[0] for p in pixels]

        # Normalize red values by shifting to upper section
        for i in range(len(h)):
            if h[i] < 30:
                h[i] = 179 - h[i]

        h_avg = np.average(h)

        if 30 <= h_avg < 90:
            return 'g'
        elif 90 <= h_avg < 150:
            return 'b'
        else:
            return 'r'

    def _join_segments(self, segments, terminals):
        """Figure out how all the segments should join, then return all simplified lines"""
        if DEBUG_PRINT:
            print('COMPONENT')
            for index, seg in enumerate(segments):
                print(f'-seg{index}', seg[0], seg[1], seg[-2], seg[-1], len(seg))

        # Tracking containers
        result = []
        terminals = list(terminals)
        colored_segments = defaultdict(list)  # color: [segment]

        # Tag each segment with a colour
        for seg in segments:
            colored_segments[self._color(seg)].append(seg)

        # For each group of segments by colour, construct a graph
        for color, segments in colored_segments.items():
            result.extend(self._process_colored_segments(color, segments, terminals))

        # Output format must be [([points], closed, color)]
        return result

    @staticmethod
    def _process_colored_segments(color, segments, terminals):
        result = []  # [([points], closed, color)]
        graph = defaultdict(list)  # (index, term?): [(index, term?)]
        removed = set()  # for single loops and removed loops
        chosen_cycles = []

        # Generate the graph by looking for segment intersections
        for si, seg in enumerate(segments):
            if seg[0] == seg[-1]:
                removed.add(si)
                chosen_cycles.append([(si, False)])
                continue
            for ti, term in enumerate(terminals):
                if seg[0] == term or seg[-1] == term:
                    graph[(ti, True)].append((si, False))
                    graph[(si, False)].append((ti, True))

        # Get out the largest cycles
        if DEBUG_PRINT:
            print('GRAPH', graph)

        def path_len(path):
            total = 0
            for node, term in path:
                if not term:
                    total += len(segments[node])
                else:
                    total -= 1
            return total

        cycles = sorted([c for c in simple_cycles(graph) if len(c) > 2], key=path_len)
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

        if DEBUG_PRINT:
            print('CYCLES', chosen_cycles)
            print('GRAPH', graph)
            print('REMOVED', removed)

        # Grab all possible paths
        found_paths = []
        for si in range(len(segments)):
            if si in removed: continue
            found_paths.extend(all_paths((si, False), graph))
            found_paths.append([(si, False)])

        found_paths.sort(key=path_len)

        if DEBUG_PRINT:
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

        if DEBUG_PRINT:
            print('FINAL CYCLES', chosen_cycles)
            print('FINAL PATHS', chosen_paths)

        def final_process(group, closed):
            # Group is of format [(si, term)]
            partial = []
            if len(group) == 1:  # Self loop or isolated
                partial = segments[group[0][0]]
            else:
                for i, (si, term) in enumerate(group):
                    if term: continue
                    segment = segments[si]
                    if i == len(group) - 1:  # If we're the last, we need to use previous
                        prev_term = terminals[group[i - 1][0]]
                        should_reverse = segment[0] != prev_term
                    else:
                        next_term = terminals[group[i + 1][0]]
                        should_reverse = segment[-1] != next_term
                    partial += reversed(segment) if should_reverse else segment

            # Be sure to approximate the path - not every pixel really matters
            result.append((cv2.approxPolyDP(np.asarray(partial), EPSILON, closed).squeeze(), closed, color))

        for cycle in chosen_cycles:
            final_process(cycle, True)
        for path in chosen_paths:
            final_process(path, False)

        return result

    def get_all_lines(self):
        """Yield out all the labelled lines, ready to be put into JSON"""
        visited = set()

        def exhaust(y_s, x_s):
            """Exhaust an entire component to extract partial segments"""
            segments = []
            branch_points = [(y_s, x_s)]  # To keep track of unfinished branches / joins
            terminals = set()  # To keep track of all branch points / joins
            comp_pixels = [0]  # To keep track of total component size. Have to keep in list to assign, gross

            def neighbors(pix):
                """Get all neighbours that need to be visited"""
                unvisit = []
                term = []
                # Check neighbour directions
                for d in NEIGH:
                    n_y, n_x = pix[0] + d[0], pix[1] + d[1]
                    if self.thinned[n_y, n_x]:
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
                    if self.thinned[n_y, n_x]:
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

                current_segment = []  # We build up segments

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
        for y in range(self.thinned.shape[0]):
            for x in range(self.thinned.shape[1]):
                if (y, x) in visited:
                    continue
                elif self.thinned[y, x]:
                    segments, terminals = exhaust(y, x)
                    # Segments is none if the total shape is too small
                    if segments is not None:
                        # Yield out all the data from the lines
                        lines = self._join_segments(segments, terminals)
                        for points, closed, color in lines:
                            yield {
                                'color': color,
                                'closed': closed,
                                'points': [{'x': int(x), 'y': int(y)} for y, x in points]
                                }
