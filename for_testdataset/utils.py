def printsave(save_path, message):
    print(message)
    with open(save_path, 'a') as f:
        f.write(message + '\n')
    return True