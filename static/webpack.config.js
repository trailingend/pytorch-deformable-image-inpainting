const webpack = require('webpack');

const config = {
    entry:  __dirname + '/script/index.jsx',
    output: {
        path: __dirname + '/dist',
        publicPath: '/dist/',
        filename: 'bundle.js'
    },
    module: {
        rules: [
            {
                test: /\.jsx?/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader']
            },
            {
                use : 'file-loader?name=[name].[ext]',
                test: /\.(jpe?g|png|gif|svg|mp4)$/
            }
            // {
            //     test: /\.html$/,
            //     use: 'html-loader?attrs[]=video:src'
            // },
            // {
            //     test: /\.mp4$/,
            //     use: 'url-loader?name=[name].[ext]&limit=10000&mimetype=video/mp4'
            // }
        ]
    },
    resolve: {
        extensions: ['.js', '.jsx', '.css']
    }
};
module.exports = config;
