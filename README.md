We are currently truing to use both pyimagej and fijibin to see which one is more realiable and helpful in our case. 

Upon finalizing the choice between fijibin and pyimageJ we will keep either in the requirements file.

To make sure the environment is set correctly you can run ImageJ.doctor.checkup()

you might need to install maven using conda instapp mvnd and you migth need to reference the path of Maven and Java_Home
in your environment variables.

It will require initializing imageJ with fiji to be able to use plugins.

Make sure you reference the fiji app and not imagej.exe

We will have to also refer to the plugin ThunderSTORM using maven instead of regular installation https://maven.scijava.org/#nexus-search;classname~Thunderstorm. More on this process here: https://forum.image.sc/t/installing-imagej-plugins-with-python/68368/2

You can include weather you would like to save the protocol alongside each image analysis or not

User can specify the file extension