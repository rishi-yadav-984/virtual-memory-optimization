# vm_simulator.py
import math
from collections import deque, OrderedDict, defaultdict
import copy
import csv

class Frame:
    def __init__(self):
        self.process = None   # owning process id
        self.page = None      # page number
        self.ref = 0          # for clock algorithm (reference bit)

    def set(self, pid, page):
        self.process = pid
        self.page = page
        self.ref = 1

    def clear(self):
        self.process = None
        self.page = None
        self.ref = 0

    def __repr__(self):
        if self.page is None:
            return "Empty"
        return f"P{self.process}:Pg{self.page}"

class VMSimulator:
    def __init__(self, num_frames, processes_refs, algorithm='LRU', demand_paging=True):
        """
        num_frames: total physical frames (int)
        processes_refs: dict {pid: [page sequence]}
        algorithm: 'LRU', 'FIFO', 'Optimal', 'Clock'
        demand_paging: boolean
        """
        self.num_frames = num_frames
        self.processes_refs = processes_refs
        self.algorithm = algorithm
        self.demand_paging = demand_paging

        # physical frames
        self.frames = [Frame() for _ in range(num_frames)]

        # page tables: dict pid -> dict page -> (frame_index or None)
        self.page_tables = defaultdict(dict)

        # structures for replacement algorithms
        self.fifo_queue = deque()           # store (pid, page)
        self.lru_order = OrderedDict()      # store (pid,page): last_time implicitly via order
        self.clock_ptr = 0                  # pointer index for clock
        # timeline: list of dicts {step, pid, page, fault, frames_snapshot}
        self.timeline = []

    def find_frame_holding(self, pid, page):
        """Return frame index if this page of pid is in frames, else None"""
        for i, f in enumerate(self.frames):
            if f.page is not None and f.process == pid and f.page == page:
                return i
        return None

    def load_page_into_frame(self, pid, page, current_time, next_uses=None):
        """Load given page of pid into a free frame or evict per algorithm"""
        # check for free frame
        for i, fr in enumerate(self.frames):
            if fr.page is None:
                fr.set(pid, page)
                self.page_tables[pid][page] = i
                # update algo structures
                self._algo_on_load(pid, page, i, current_time)
                return i, True  # frame index, was_free
        # need eviction
        victim_idx = self._select_victim(pid, page, current_time, next_uses)
        victim_frame = self.frames[victim_idx]
        # evict victim
        old_pid, old_page = victim_frame.process, victim_frame.page
        # remove from page table
        if old_page in self.page_tables.get(old_pid, {}):
            self.page_tables[old_pid][old_page] = None
        # place new page
        victim_frame.set(pid, page)
        self.page_tables[pid][page] = victim_idx
        # update structures post eviction
        self._algo_on_load(pid, page, victim_idx, current_time)
        return victim_idx, False

    def _algo_on_load(self, pid, page, frame_idx, current_time):
        key = (pid, page)
        if self.algorithm == 'LRU':
            # move to end as most recently used
            if key in self.lru_order:
                del self.lru_order[key]
            self.lru_order[key] = current_time
        elif self.algorithm == 'FIFO':
            # add to FIFO queue
            self.fifo_queue.append(key)
        elif self.algorithm == 'Clock':
            # set reference bit
            self.frames[frame_idx].ref = 1
        # Optimal does not need structure on load (lookahead used during selection)

    def _select_victim(self, pid, page, current_time, next_uses):
        """Decide which frame index to evict based on algorithm"""
        if self.algorithm == 'LRU':
            # least recently used -> first item in OrderedDict
            if not self.lru_order:
                return 0
            oldest_key = next(iter(self.lru_order))
            for i, f in enumerate(self.frames):
                if f.process == oldest_key[0] and f.page == oldest_key[1]:
                    del self.lru_order[oldest_key]
                    return i
            return 0

        elif self.algorithm == 'FIFO':
            while self.fifo_queue:
                key = self.fifo_queue.popleft()
                for i, f in enumerate(self.frames):
                    if f.process == key[0] and f.page == key[1]:
                        return i
            return 0

        elif self.algorithm == 'Optimal':
            farthest = None
            farthest_use = -1
            for i, f in enumerate(self.frames):
                key = (f.process, f.page)
                nu = next_uses.get(key, math.inf)
                if nu == math.inf:
                    return i
                if nu > farthest_use:
                    farthest_use = nu
                    farthest = i
            return farthest if farthest is not None else 0

        elif self.algorithm == 'Clock':
            n = self.num_frames
            while True:
                frame = self.frames[self.clock_ptr]
                if frame.ref == 0:
                    victim = self.clock_ptr
                    self.clock_ptr = (self.clock_ptr + 1) % n
                    return victim
                else:
                    frame.ref = 0
                    self.clock_ptr = (self.clock_ptr + 1) % n
        else:
            return 0

    def simulate_global_trace(self, global_ref_string):
        """Simulate when you give a single global reference string of (pid,page) pairs."""
        self._reset_state()
        time = 0
        total_faults = 0
        # precompute positions for Optimal
        future_positions = defaultdict(list)
        for idx, (pid, page) in enumerate(global_ref_string):
            future_positions[(pid, page)].append(idx)

        for idx, (pid, page) in enumerate(global_ref_string):
            # pop current index from future list
            future_positions[(pid, page)].pop(0)
            # prepare next_uses mapping for frames for Optimal
            next_uses = {}
            if self.algorithm == 'Optimal':
                for f in self.frames:
                    if f.page is not None:
                        key = (f.process, f.page)
                        lst = future_positions.get(key, [])
                        next_uses[key] = lst[0] if lst else math.inf

            in_frame_idx = self.find_frame_holding(pid, page)
            fault = False
            if in_frame_idx is None:
                total_faults += 1
                fault = True
                self.load_page_into_frame(pid, page, idx, next_uses)
            else:
                if self.algorithm == 'LRU':
                    k = (pid, page)
                    if k in self.lru_order:
                        del self.lru_order[k]
                    self.lru_order[k] = idx
                if self.algorithm == 'Clock':
                    self.frames[in_frame_idx].ref = 1

            snap = [repr(fr) for fr in self.frames]
            self.timeline.append({
                'step': idx,
                'pid': pid,
                'page': page,
                'fault': fault,
                'frames': snap
            })
            time += 1

        return {'page_faults': total_faults, 'timeline': self.timeline}

    def simulate_interleaved_processes(self):
        """Create a global interleaved timeline from per-process reference streams."""
        lists = {pid: list(seq) for pid, seq in self.processes_refs.items()}
        global_ref = []
        while any(lists.values()):
            for pid in sorted(lists.keys()):
                if lists[pid]:
                    global_ref.append((pid, lists[pid].pop(0)))
        return self.simulate_global_trace(global_ref)

    def _reset_state(self):
        self.frames = [Frame() for _ in range(self.num_frames)]
        self.page_tables = defaultdict(dict)
        self.fifo_queue = deque()
        self.lru_order = OrderedDict()
        self.clock_ptr = 0
        self.timeline = []

    def pretty_print_timeline(self):
        print("Step | PID | Page | Fault | Frames")
        print("----------------------------------------------")
        for entry in self.timeline:
            print(f"{entry['step']:3d}  | {entry['pid']:3d} | {entry['page']:4} | {'Y' if entry['fault'] else 'N'}   | {entry['frames']}")

    def export_timeline_csv(self, fname):
        headers = ['step','pid','page','fault'] + [f'frame_{i}' for i in range(self.num_frames)]
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for e in self.timeline:
                row = [e['step'], e['pid'], e['page'], int(e['fault'])] + e['frames']
                writer.writerow(row)
        print(f"Exported timeline to {fname}")

if __name__ == '__main__':
    processes = {
        1: [1, 2, 3, 2, 4, 1, 5],
        2: [2, 3, 2, 1, 4]
    }
    print("Simulator example: 3 frames, algorithm=LRU")
    sim = VMSimulator(num_frames=3, processes_refs=processes, algorithm='LRU')
    result = sim.simulate_interleaved_processes()
    print("Page faults:", result['page_faults'])
    sim.pretty_print_timeline()
    sim2 = VMSimulator(num_frames=3, processes_refs=processes, algorithm='Optimal')
    res2 = sim2.simulate_interleaved_processes()
    print("\nOptimal Page faults:", res2['page_faults'])
    sim2.pretty_print_timeline()
