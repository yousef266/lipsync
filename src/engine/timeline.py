from typing import List, Optional, Dict
from ..core.models import TimedValue
from ..linguistics.mapper import ArabicShapeMapper

class Timeline:
    """Timeline with smooth transitions"""

    def __init__(self):
        self.elements: List[TimedValue] = []

    def add(self, start: int, end: int, value: any, metadata: Dict = None):
        """Add a timed value"""
        self.elements.append(TimedValue(start, end, value, metadata or {}))

    def get_at(self, time_cs: int) -> Optional[any]:
        """Get value at specific time"""
        for element in self.elements:
            if element.start <= time_cs < element.end:
                return element.value
        return None

    def optimize(self):
        """Merge adjacent elements with same value"""
        if not self.elements:
            return

        self.elements.sort(key=lambda x: x.start)
        optimized = [self.elements[0]]

        for element in self.elements[1:]:
            last = optimized[-1]
            if last.value == element.value and last.end == element.start:
                last.end = element.end
                last.metadata.update(element.metadata)
            else:
                optimized.append(element)

        self.elements = optimized

    def add_tweening(self, min_tween_duration_cs: int = 4):
        """Add smooth transitions between shapes"""
        if len(self.elements) < 2:
            return

        new_elements = [self.elements[0]]

        for i in range(1, len(self.elements)):
            prev = new_elements[-1]
            curr = self.elements[i]

            tween_shape = ArabicShapeMapper.get_tween_shape(prev.value, curr.value)

            if tween_shape and curr.start - prev.end < 1:
                # Add tween
                tween_duration = min(
                    min_tween_duration_cs, (curr.end - prev.start) // 3
                )

                if tween_duration >= min_tween_duration_cs:
                    tween_start = curr.start
                    tween_end = curr.start + tween_duration

                    new_elements.append(
                        TimedValue(tween_start, tween_end, tween_shape, {"tween": True})
                    )

                    curr.start = tween_end

            new_elements.append(curr)

        self.elements = new_elements
