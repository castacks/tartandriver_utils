from builtin_interfaces.msg import Time

def stamp_to_time(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def time_to_stamp(t):
    sec = int(t)
    nsec = int((t-sec)*1e9)

    return Time(sec=sec, nanosec=nsec)