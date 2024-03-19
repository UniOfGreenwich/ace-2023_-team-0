# **Networking with nm-connection-editor**
 
## **1. Background of nm-connection-editor**

The nm-connection-editor helps to create new network connections and manage existing network connections in Linux OS. It is a GUI based application.

                sudo nm-connection-editor

By typing the above command in Linux terminal, user can start the nm connection editor application. It is only available in GUI (desktop or GNOME) environment. To instead work with terminal, can use nm-cli connection. It available in all platforms such as terminal, desktop and ssh,or instead, can use nmtui, it is a curses-based application. It works on terminal and desktop and support all the feature of network manager.

<b>Table 1: Comparing nmtui and nm-connection-editor</b>
![nmtui_nmconnections](nm-connection-nmtui.png)<br>
<br><br>

## **2. Methodology of nm-connection-editor**
 
### **2.1 How to add a new network connection with nm-connection-editor**
---
<br>

![new_nm_connections](add-connection.png)<br>
<b>Figure 1:Flow diagram to add a new network connection in nm-connection-editor</b>
<br><br>


In the head node, the nm-connection editor saves all the network configuration files in etc/sysconfig/network-scripts/ directory.

                    sudo nmcli connection show

By using the above command, can verify the made network connections.

                    sudo nmcli connection up [Network Connection name]
By using the above command, can activate the specific network connection.

                    sudo ip address show [Network interface]

After activating network connection, can view the IP configuration of the specific device.
<br>

## **2.2 How to edit a new network connection with nm-connection-editor**
---

![edit_nm_connections](edit-connection.png)<br>
<b>Figure 2:Flow diagram to edit network connection in nm-connection-editor</b>
<br><br>


                    sudo nmcli connection down [Network Connection name]
		            sudo nmcli connection up [Network Connection name]
After customisation, the Network Manager/ Net plan won’t change the customise options. Therefore, must manually change the customisation by shutting down and turning on network connection.

                    sudo ip address show [Network Interface]
Verify the customised  IP address changed/ updated.
<br>

 ### **2.3 How to delete a new network connection with nm-connection-editor**
 ---
 
![delete_nm_connections](delete-connection.png)<br>
<b>Figure 3:Flow diagram to delete a network connection in nm-connection-editor</b>
<br><br>

                    sudo nmcli connection show 
Can verify the connection above was deleted.
<br>


## **3. Implementation of nm-connection-editor**





 ## **4. Helpful Resources**
 ---
1. How to implement nm-connection editor: https://help.ubuntu.com/community/NetworkManager



