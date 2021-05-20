import axios from 'axios'

const http = axios.create({
    // http://127.0.0.1:65318/
    baseURL: 'http://127.0.0.1:5000/'
    // baseURL: 'http://localhost:5000/'
})

export default http