var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var fs = require('fs');

app.use(express.static(__dirname)); // this will serve your static files

app.get('/', function(req, res){
    res.sendFile(__dirname + '/index.html');
});

io.of('/kizuna').on('connection', (socket) => {
    console.log('a kizuna client connected');

    socket.on('result_data', (result) => {
        if (result != 0) {
            socket.broadcast.emit('result_download', result);
        }
    });

    socket.on('disconnect', () => { console.log('a kizuna client disconnected') });
});

http.listen(6789, () => console.log('listening on http://127.0.0.1:6789/kizuna.html'));
