import time
import paramiko


# File used to connect and communicate with EV3 brick
def establish_ssh_connection():
    # Connect to the robot via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    your_username = 'robot'
    your_password = 'maker'
    your_remote_host = 'ev3dev.local'
    ssh.connect(your_remote_host, username=your_username, password=your_password)

    # Open a new channel
    channel = ssh.invoke_shell()

    channel.send("cd ev3dev-c/workplace\n")
    time.sleep(1)

    # Run the C code
    channel.send("./fullcode\n")
    time.sleep(1)

    return ssh, channel


def send_channel_commands(channel, move):

    # Send the command
    channel.send(str(move) + '\n')
    time.sleep(1)

    # Receive output and display output
    output = ""
    while not channel.recv_ready():
        time.sleep(0.5)

    while channel.recv_ready():
        output += channel.recv(4096).decode()

    print(f'Output:\n{output}')




