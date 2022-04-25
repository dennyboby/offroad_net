LINE = "".join(["-"] * 100)


def print_dict(dict_x, dict_name="dictionary"):
    str_report = f"{LINE}\n{dict_name}:{len(dict_x)}\n{LINE}"
    for key, val in dict_x.items():
        str_report += f"\n{key}: {val}"
    str_report += f"\n{LINE}\n"
    return str_report
