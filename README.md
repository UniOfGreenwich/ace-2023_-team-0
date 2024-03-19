# 1. Wake-On-LAN for power up HPC

## **1.1 Introduction**
 ---
- Aim : Introducing a power on button to HPC.
- It will help to turn on and off the HPC.
- Objectives : Hardware Design, Network Configuration, WOL (Wake On LAN) 
<br><br>


 ## **1.2 Methodology**
 ---
1. How to connect a power button to motherboard?
    - Finding apporpraiate power button (momentary switch)
    - Finding the header for power button in motherboard (Front panel connector)
    - Configure BIOS setting ( In power management setting)

2. How to connect compute nodes and head node to enable the power buttons? (Hardware Design)
    - Needed components : Switch with maximum 11 ports (1 for header node and 10 for compute node) and ethernet cable (11 is enough)
    - Have to crimple ethernet cables following CAT 5E standards
    - Building appopriate node connections and power in motherboard
    - Network Configuration (Done through hard coded or DHCP)
    - Verfify the connections (Using ping command)
    - Mostly, in compute nodes don't need the front panel connection for button expect if want to reset 

3. How to enable power on button between head node and compute nodes? (Software design)
    - Enabling WOL (Wake On LAN) software package
    - Install WOL package
    - Get MAC address           (Same to what we are doing in network booting)
    - Send WOL packets to compute nodes
<br><br>


 ## **1.3 Implementation**
 ---

 

 # **2. Networking with nm-connection-editor**
 
## **2.1 Background of nm-connection-editor**

The nm-connection-editor helps to create new network connections and manage existing network connections in Linux OS. It is a GUI based application.

                sudo nm-connection-editor

By typing the above command in Linux terminal, user can start the nm connection editor application. It is only available in GUI (desktop or GNOME) environment. To instead work with terminal, can use nm-cli connection. It available in all platforms such as terminal, desktop and ssh,or instead, can use nmtui, it is a curses-based application. It works on terminal and desktop and support all the feature of network manager.

Table 1: Comparing nmtui and nm-connection-editor

## **2.2 Methodology of nm-connection-editor**
 How to add a new network connection with nm-connection-editor**
---

![add nm-connection](add-nm-connection.png)

Figure 1: Comparing nmtui and nm-connection-editor
<br>

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

![add nm-connection](add-nm-connection.png)

Figure 2: Comparing nmtui and nm-connection-editor
<br>

                    sudo nmcli connection down [Network Connection name]
		            sudo nmcli connection up [Network Connection name]
After customisation, the Network Manager/ Net plan won’t change the customise options. Therefore, must manually change the customisation by shutting down and turning on network connection.

                    sudo ip address show [Network Interface]
Verify the customised  IP address changed/ updated.
<br>

 ### **2.3 How to delete a new network connection with nm-connection-editor**
 ---

 ![add nm-connection](add-nm-connection.png)

Figure 3: Comparing nmtui and nm-connection-editor
<br>

                    sudo nmcli connection show 
Can verify the connection above was deleted.
<br>








 ### **4. Helpful Resources**
 ---
1. The button can be find here: https://uk.rs-online.com/web/p/push-button-switches/2099127?cm_mmc=UK-PLA-DS3A-_-google-_-CSS_UK_EN_PMAX_RS+PRO-_--_-2099127&matchtype=&&gad_source=1&gclid=EAIaIQobChMIpP3xg7-mhAMVFAUGAB0t5QD5EAQYByABEgIy1_D_BwE&gclsrc=aw.ds


2. How to install WOL : https://pimylifeup.com/ubuntu-enable-wake-on-lan/#:~:text=Wake%2Don%2DLAN%20is%20a,functionality%20through%20your%20devices%20BIOS.

3. Power button and Front panel connections : https://www.pcinq.com/how-to-connect-motherboard-front-panel-headers/
                            https://www.electronicshub.org/power-button-on-motherboard/#:~:text=Ans%3A%20The%20power%20switch%20on,for%20the%2020-pin%20header
<br><br>


## Authors

 ### **5. Images**
 ---
![Push button](PushButton.png)<br>
<b>Figure 1:Push button to be used in the HPC</b>
<br><br>


![Wake On LAN setup](WakeonLan.png)<br>
<b>Figure 2:Wake On LAN</b>
<br>

