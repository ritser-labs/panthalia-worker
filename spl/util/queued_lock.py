import threading
import queue
import itertools

class QueuedLock:
    def __init__(self):
        self._queue = queue.PriorityQueue()  # PriorityQueue to store tasks with priority based on time_status_changed
        self._lock = threading.Lock()  # Internal lock for synchronizing access to the shared resource
        self._is_locked = False  # Track whether the lock is currently held
        self._counter = itertools.count()  # Unique counter to ensure uniqueness in the priority queue

    def acquire(self, priority):
        """
        Acquire the lock for a task with a given priority.
        :param priority: The priority (lower values get higher priority, e.g., time_status_changed)
        """
        # Create an event for this thread and put it in the priority queue with a unique counter
        event = threading.Event()
        count = next(self._counter)
        self._queue.put((priority, count, event))  # Add tuple (priority, count, event)

        # Only the thread with the highest priority (lowest time_status_changed) can acquire the lock
        while True:
            if self._queue.queue[0][2] == event:  # Check if the current thread is at the front (highest priority)
                with self._lock:
                    if not self._is_locked:
                        self._is_locked = True
                        return
            event.wait()  # Wait until the event is set

    def release(self):
        """
        Release the lock, allowing the next task in the queue (based on priority) to proceed.
        """
        with self._lock:
            if not self._is_locked:
                raise RuntimeError("Cannot release an unlocked QueuedLock")
            self._is_locked = False

            # Remove the event from the front of the priority queue
            priority, count, event = self._queue.get()
            if not self._queue.empty():
                # Notify the next thread in the queue (with the highest priority)
                next_event = self._queue.queue[0][2]
                next_event.set()
