/*
 # The position calculation server based on ZeroMQ platform
 # CSRHOS - Chinese Solar Radio HelioGraph Operation System
 #
 # Created: Since 2013-1-1
 # Shoulin Wei Feng Wang @ CSRHOS Team
 #
 # This file is part of CSRH project
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <czmq.h>
#define Server_EndPoint    "tcp://127.0.0.1:5555"
#define REQUEST_TIMEOUT     2000
#define REQUEST_RETRIES     3
int main (int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: ./%s Loop year-month-day-hour-target\n",argv[0]);
        exit(1);
    }
    zctx_t*context = zctx_new ();
    //printf ("U: Hello,client is connecting to ephem server...\n");
    void *client = zsocket_new (context, ZMQ_REQ);
    assert (client);
    char date[30];
    sprintf (date, "%s", argv[2]);
    zsocket_connect (client,Server_EndPoint);
    int sequence = 0;
    int retries_left = REQUEST_RETRIES;
    int request_left = atoi(argv[1]);

    srandom ((unsigned) time (NULL));
    while (retries_left && !zctx_interrupted && request_left) {

        //double testTime = (double)randof (23)+(double)randof (59)/60+(double)randof (59)/3600+(double)randof (999)/3600000;
        //sprintf (date, "%lf", testTime);
        int64_t clock = zclock_time ();
        zstr_send (client, date);

        int expect_reply = 1;
        while (expect_reply) {
            zmq_pollitem_t items [] = { { client, 0, ZMQ_POLLIN, 0 } };
            int rc = zmq_poll (items, 1, REQUEST_TIMEOUT * ZMQ_POLL_MSEC);
            if (rc == -1)
                break;          //  Interrupted

            if (items [0].revents & ZMQ_POLLIN) {
                //  We got a reply from the server, must match sequence
                char *reply = zstr_recv (client);
                if (!reply)
                    break;      //  Interrupted
				retries_left = REQUEST_RETRIES;
				expect_reply = 0;
                printf("response:%s\n",reply);
                //printf ("%d,%dms\n", request_left--,zclock_time () - clock);
                free (reply);
            }
            else if (--retries_left == 0) {
               // printf ("U: Sorry,ephem server seems to be offline, abandoning...\n");
                break;
            }
            else {
                //printf ("U: Sorry,no response from ephem server, retrying...\n");
                //  Old socket is confused; close it and open a new one
                zsocket_destroy (context, client);
                //printf ("U: Hello,reconnecting to ephem server…\n");
                client = zsocket_new (context, ZMQ_REQ);
                zsocket_connect (client, Server_EndPoint);
                //  Send request again, on new socket
                zstr_send (client, date);
                char *response= zstr_recv(client);
               // printf("Response:%s\n",response);
            }
        }
		zclock_sleep (100);
    }
    zctx_destroy (&context);
    return 0;
}

