import threading
import queue

class QueuedLock:
    def __init__(self):
        self._queue = queue.Queue()  # Queue to store the order of lock requests
        self._lock = threading.Lock()  # Internal lock for synchronizing access to the shared resource
        self._is_locked = False  # Track whether the lock is currently held

    def acquire(self):
        # Create an event for this thread and put it in the queue
        event = threading.Event()
        self._queue.put(event)

        # Only the thread at the front of the queue can acquire the lock
        while True:
            if self._queue.queue[0] == event:  # Check if the current thread is at the front
                with self._lock:
                    if not self._is_locked:
                        self._is_locked = True
                        return
            event.wait()  # Wait until the event is set

    def release(self):
        with self._lock:
            if not self._is_locked:
                raise RuntimeError("Cannot release an unlocked QueuedLock")
            self._is_locked = False

            # Remove the event from the front of the queue
            event = self._queue.get()
            if not self._queue.empty():
                # Notify the next thread in the queue
                self._queue.queue[0].set()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()