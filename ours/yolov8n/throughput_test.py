# Auto-generated test script for the custom operator on PYNQ

import numpy as np
from pynq import Overlay
from pynq import allocate
from pynq import PL
import time

PL.reset()
ol = Overlay("Overlay/design.bit")
kernel = ol.yolov8n_0

def write_u64(ip, offset, value):
    ip.write(offset, value & 0xFFFFFFFF)
    ip.write(offset + 4, (value >> 32) & 0xFFFFFFFF)


FRAMES = 300
RING_DEPTH = 3
INPUT_BUFFERS = RING_DEPTH
OUTPUT_BUFFERS = RING_DEPTH
POLL_SLEEP_S = 0.0005

def release_buffer(buf):
    freebuffer = getattr(buf, "freebuffer", None)
    if freebuffer is not None:
        freebuffer()

def release_buffer_list(buffers):
    for buf in buffers:
        release_buffer(buf)
    buffers.clear()

DMA_MM2S = 1
DMA_S2MM = 0
DMACR_RS = 0x00000001
DMACR_RESET = 0x00000004
DMASR_HALTED = 0x00000001
DMASR_IDLE = 0x00000002
DMASR_ERROR_MASK = 0x00000770
BD_STS_COMPLETE = 1 << 31
BD_STS_ERROR_MASK = 0x30000000
BD_CTRL_SOF = 1 << 27
BD_CTRL_EOF = 1 << 26
BD_CTRL_LEN_MASK = 0x03FFFFFF

class SgDmaRing:
    def __init__(self, dma_ip, direction, depth, name):
        self.mmio = dma_ip.mmio
        self.direction = direction
        self.depth = depth
        self.name = name
        self.offset = 0x00 if direction == DMA_MM2S else 0x30
        self.flush_before = direction == DMA_MM2S
        self.desc = allocate(shape=(depth, 16), dtype=np.uint32)
        self.free = list(range(depth))
        self.queued = []
        self.started = False
        self.tail = None
        self._init_descriptors()
        self.reset()

    def _reg(self, off):
        return self.offset + off

    def _desc_addr(self, idx):
        return self.desc.physical_address + idx * 16 * 4

    def _init_descriptors(self):
        self.desc[:] = 0
        for idx in range(self.depth):
            next_addr = self._desc_addr((idx + 1) % self.depth)
            self.desc[idx, 0] = next_addr & 0xFFFFFFFF
            self.desc[idx, 1] = (next_addr >> 32) & 0xFFFFFFFF
        self.desc.flush()

    def reset(self):
        self.mmio.write(self._reg(0x00), DMACR_RESET)
        while self.mmio.read(self._reg(0x00)) & DMACR_RESET:
            pass
        self.mmio.write(self._reg(0x04), 0xFFFFFFFF)
        self.mmio.write(self._reg(0x00), 0x00000000)
        self.started = False
        self.tail = None

    def has_free(self):
        return bool(self.free)

    def status(self):
        return self.mmio.read(self._reg(0x04))

    def _check_error(self):
        sr = self.status()
        if sr & DMASR_ERROR_MASK:
            raise RuntimeError(f"{self.name} SG DMA error: status=0x{sr:08x}")

    def _start_at(self, idx):
        addr = self._desc_addr(idx)
        self.mmio.write(self._reg(0x08), addr & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x0C), (addr >> 32) & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x00), DMACR_RS)
        while self.mmio.read(self._reg(0x04)) & DMASR_HALTED:
            self._check_error()
        self.started = True

    def enqueue(self, seq, buf, nbytes=None):
        if not self.free:
            raise RuntimeError(f"{self.name}: no free SG descriptors")
        if nbytes is None:
            nbytes = buf.nbytes
        if nbytes <= 0 or nbytes > BD_CTRL_LEN_MASK:
            raise ValueError(f"{self.name}: invalid transfer length {nbytes}")

        idx = self.free.pop(0)
        addr = buf.physical_address
        self.desc[idx, 2] = addr & 0xFFFFFFFF
        self.desc[idx, 3] = (addr >> 32) & 0xFFFFFFFF
        self.desc[idx, 4] = 0
        self.desc[idx, 5] = 0
        self.desc[idx, 6] = (nbytes & BD_CTRL_LEN_MASK) | BD_CTRL_SOF | BD_CTRL_EOF
        self.desc[idx, 7] = 0
        self.desc[idx, 8:16] = 0
        self.desc.flush()

        if self.flush_before:
            buf.flush()

        if not self.started:
            self._start_at(idx)

        tail_addr = self._desc_addr(idx)
        self.mmio.write(self._reg(0x10), tail_addr & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x14), (tail_addr >> 32) & 0xFFFFFFFF)
        self.tail = idx
        self.queued.append({"seq": seq, "idx": idx, "buf": buf, "nbytes": nbytes})
        return self.queued[-1]

    def poll(self):
        self._check_error()
        completed = []
        while self.queued:
            head = self.queued[0]
            idx = head["idx"]
            self.desc.invalidate()
            status = int(self.desc[idx, 7])
            if not (status & BD_STS_COMPLETE):
                break
            if status & BD_STS_ERROR_MASK:
                raise RuntimeError(f"{self.name}: descriptor {idx} error status=0x{status:08x}")
            if not self.flush_before:
                head["buf"].invalidate()
            completed.append(self.queued.pop(0))
        return completed

    def reclaim(self, desc_info):
        idx = desc_info["idx"]
        self.desc[idx, 7] = 0
        self.desc.flush()
        self.free.append(idx)

    def close(self):
        self.reset()
        release_buffer(self.desc)

global_in_buffers = [allocate(shape=(640, 640, 3), dtype="int8") for _ in range(INPUT_BUFFERS)]
for _buf in global_in_buffers:
    _buf[:] = np.random.randint(-128, 127, size=(640, 640, 3), dtype="int8")
global_in_1_buffer = allocate(shape=(789399,), dtype="uint32")
global_in_1_data = np.random.randint(0, 4294967295, size=(789399,), dtype="uint32")
global_out_buffers = [allocate(shape=(8400, 80), dtype="int8") for _ in range(OUTPUT_BUFFERS)]
global_out_1_buffers = [allocate(shape=(8400, 4, 1), dtype="float32") for _ in range(OUTPUT_BUFFERS)]
StreamingTensorDuplicator_6_out1_buffer = allocate(shape=(409600,), dtype=np.int8)
StreamingTensorDuplicator_6_out1_buffer[:] = 0
StreamingTensorDuplicator_6_out1_addr = StreamingTensorDuplicator_6_out1_buffer.device_address
write_u64(kernel, 0x10, StreamingTensorDuplicator_6_out1_addr)
write_u64(kernel, 0x1C, StreamingTensorDuplicator_6_out1_addr)
StreamingConcat_26_out0_buffer = allocate(shape=(1152000,), dtype=np.int8)
StreamingConcat_26_out0_buffer[:] = 0
StreamingConcat_26_out0_addr = StreamingConcat_26_out0_buffer.device_address
write_u64(kernel, 0x28, StreamingConcat_26_out0_addr)
write_u64(kernel, 0x34, StreamingConcat_26_out0_addr)
BandwidthAdjustDecreaseWord_26_out0_buffer = allocate(shape=(230400,), dtype=np.int8)
BandwidthAdjustDecreaseWord_26_out0_buffer[:] = 0
BandwidthAdjustDecreaseWord_26_out0_addr = BandwidthAdjustDecreaseWord_26_out0_buffer.device_address
write_u64(kernel, 0x40, BandwidthAdjustDecreaseWord_26_out0_addr)
write_u64(kernel, 0x4C, BandwidthAdjustDecreaseWord_26_out0_addr)
StreamingTensorDuplicator_11_out1_buffer = allocate(shape=(204800,), dtype=np.int8)
StreamingTensorDuplicator_11_out1_buffer[:] = 0
StreamingTensorDuplicator_11_out1_addr = StreamingTensorDuplicator_11_out1_buffer.device_address
write_u64(kernel, 0x58, StreamingTensorDuplicator_11_out1_addr)
write_u64(kernel, 0x64, StreamingTensorDuplicator_11_out1_addr)
StreamingReshape_0_out0_buffer = allocate(shape=(921600,), dtype=np.int8)
StreamingReshape_0_out0_buffer[:] = 0
StreamingReshape_0_out0_addr = StreamingReshape_0_out0_buffer.device_address
write_u64(kernel, 0x70, StreamingReshape_0_out0_addr)
write_u64(kernel, 0x7C, StreamingReshape_0_out0_addr)
BandwidthAdjustIncreaseWord_8_out0_buffer = allocate(shape=(204800,), dtype=np.int8)
BandwidthAdjustIncreaseWord_8_out0_buffer[:] = 0
BandwidthAdjustIncreaseWord_8_out0_addr = BandwidthAdjustIncreaseWord_8_out0_buffer.device_address
write_u64(kernel, 0x88, BandwidthAdjustIncreaseWord_8_out0_addr)
write_u64(kernel, 0x94, BandwidthAdjustIncreaseWord_8_out0_addr)
StreamingConv_42_out0_buffer = allocate(shape=(409600,), dtype=np.int8)
StreamingConv_42_out0_buffer[:] = 0
StreamingConv_42_out0_addr = StreamingConv_42_out0_buffer.device_address
write_u64(kernel, 0xA0, StreamingConv_42_out0_addr)
write_u64(kernel, 0xAC, StreamingConv_42_out0_addr)
BandwidthAdjustIncreaseWord_14_out0_buffer = allocate(shape=(409600,), dtype=np.int8)
BandwidthAdjustIncreaseWord_14_out0_buffer[:] = 0
BandwidthAdjustIncreaseWord_14_out0_addr = BandwidthAdjustIncreaseWord_14_out0_buffer.device_address
write_u64(kernel, 0xB8, BandwidthAdjustIncreaseWord_14_out0_addr)
write_u64(kernel, 0xC4, BandwidthAdjustIncreaseWord_14_out0_addr)
global_in_1_buffer[:] = global_in_1_data[:]
ol.global_in_1_dma.sendchannel.transfer(global_in_1_buffer)
ol.global_in_1_dma.sendchannel.wait()
print('Static input global_in_1 loaded')

input_rings = {}
output_rings = {}
input_rings['global_in'] = SgDmaRing(ol.global_in_dma, DMA_MM2S, RING_DEPTH, 'global_in_mm2s')
output_rings['global_out'] = SgDmaRing(ol.global_out_dma, DMA_S2MM, RING_DEPTH, 'global_out_s2mm')
output_rings['global_out_1'] = SgDmaRing(ol.global_out_1_dma, DMA_S2MM, RING_DEPTH, 'global_out_1_s2mm')

free_slots = list(range(RING_DEPTH))
active = {}
sent_frames = 0
completed_frames = 0
blocked_no_slot = 0
blocked_no_desc = 0
submit_times = {}
latencies_s = []

def rings_have_capacity():
    return all(ring.has_free() for ring in input_rings.values()) and all(
        ring.has_free() for ring in output_rings.values()
    )

start_s = time.perf_counter()

while completed_frames < FRAMES:
    now = time.perf_counter()

    for name, ring in input_rings.items():
        for desc in ring.poll():
            req = active.get(desc["seq"])
            if req is not None:
                req["input_done"][name] = True

    for name, ring in output_rings.items():
        for desc in ring.poll():
            req = active.get(desc["seq"])
            if req is not None:
                req["output_done"][name] = True

    for seq in sorted(list(active.keys())):
        req = active[seq]
        if all(req["output_done"].values()):
            completed_frames += 1
            submit_s = submit_times.pop(seq, None)
            if submit_s is not None:
                latencies_s.append(now - submit_s)
            for desc in req["input_descs"]:
                desc["ring"].reclaim(desc)
            for desc in req["output_descs"]:
                desc["ring"].reclaim(desc)
            free_slots.append(req["slot"])
            del active[seq]

    while sent_frames < FRAMES:
        if not free_slots:
            blocked_no_slot += 1
            break
        if not rings_have_capacity():
            blocked_no_desc += 1
            break

        slot = free_slots.pop(0)
        seq = sent_frames
        req = {
            "slot": slot,
            "input_done": {name: False for name in input_rings},
            "output_done": {name: False for name in output_rings},
            "input_descs": [],
            "output_descs": [],
        }

        desc = output_rings['global_out'].enqueue(seq, global_out_buffers[slot])
        desc['ring'] = output_rings['global_out']
        req['output_descs'].append(desc)
        desc = output_rings['global_out_1'].enqueue(seq, global_out_1_buffers[slot])
        desc['ring'] = output_rings['global_out_1']
        req['output_descs'].append(desc)
        desc = input_rings['global_in'].enqueue(seq, global_in_buffers[slot])
        desc['ring'] = input_rings['global_in']
        req['input_descs'].append(desc)

        active[seq] = req
        submit_times[seq] = now
        sent_frames += 1

    if POLL_SLEEP_S > 0:
        time.sleep(POLL_SLEEP_S)

total_s = time.perf_counter() - start_s
sorted_latencies = sorted(latencies_s)
avg_latency_ms = (sum(latencies_s) / len(latencies_s)) * 1e3 if latencies_s else float("nan")
p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2] * 1e3 if sorted_latencies else float("nan")
max_latency_ms = max(latencies_s) * 1e3 if latencies_s else float("nan")

print("===== SG streaming benchmark results =====")
print(f"Frames submitted:          {sent_frames}")
print(f"Frames completed:          {completed_frames}")
print(f"Total measured time (s):   {total_s:.6f}")
print(f"Completed throughput img/s:{completed_frames / total_s:.2f}")
print(f"Avg submit-to-output ms:   {avg_latency_ms:.3f}")
print(f"P50 submit-to-output ms:   {p50_latency_ms:.3f}")
print(f"Max submit-to-output ms:   {max_latency_ms:.3f}")
print(f"No-free-slot polls:        {blocked_no_slot}")
print(f"No-free-desc polls:        {blocked_no_desc}")

for ring in input_rings.values():
    ring.close()
for ring in output_rings.values():
    ring.close()
release_buffer_list(global_out_buffers)
release_buffer_list(global_out_1_buffers)
release_buffer_list(global_in_buffers)
release_buffer(global_in_1_buffer)
release_buffer(StreamingTensorDuplicator_6_out1_buffer)
release_buffer(StreamingConcat_26_out0_buffer)
release_buffer(BandwidthAdjustDecreaseWord_26_out0_buffer)
release_buffer(StreamingTensorDuplicator_11_out1_buffer)
release_buffer(StreamingReshape_0_out0_buffer)
release_buffer(BandwidthAdjustIncreaseWord_8_out0_buffer)
release_buffer(StreamingConv_42_out0_buffer)
release_buffer(BandwidthAdjustIncreaseWord_14_out0_buffer)
