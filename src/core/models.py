from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TimeRange:
    """Time range in centiseconds"""
    start: int
    end: int

    def duration(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class TimedValue:
    """A value with associated time range"""
    start: int
    end: int
    value: any
    metadata: Dict = field(default_factory=dict)

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start, self.end)
