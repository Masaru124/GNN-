import psutil
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_times']):
    if 'python' in proc.info['name'].lower():
        print(f"PID: {proc.info['pid']}")
        print(f"  Cmdline: {proc.info['cmdline']}")
        print(f"  Memory Working Set: {proc.info['memory_info'].rss / (1024*1024):.1f} MB")
        cpu_t = proc.info['cpu_times']
        if cpu_t:
            print(f"  CPU Time: User={cpu_t.user:.1f}s, System={cpu_t.system:.1f}s")
