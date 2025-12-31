import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Video, 
  Users, 
  MessageSquare, 
  Heart, 
  Share2, 
  MoreVertical, 
  Search, 
  Play, 
  Pause, 
  Volume2, 
  VolumeX,
  Star
} from 'lucide-react'
import { apiClient } from '@/services/api'

// 定义直播流数据类型
interface LiveStream {
  id: number
  title: string
  streamer: string
  category: string
  viewers: number
  status: 'live' | 'upcoming' | 'ended'
  cover_image: string | null
  tags: string[]
  start_time: string
  streamer_avatar: string
  description: string
}

// 初始空直播流数据
const initialLiveStreams: LiveStream[] = []



// Mock community posts data
const communityPosts = [
  {
    id: 1,
    user: 'AI爱好者小明',
    avatar: 'https://ui-avatars.com/api/?name=小明&background=random',
    content: '今天使用AI平台的模型训练功能，效果非常好！训练速度比之前提升了30%，推荐大家使用。',
    likes: 128,
    comments: 35,
    time: '2小时前',
    tags: ['模型训练', 'AI', '技术分享']
  },
  {
    id: 2,
    user: '农业科技达人',
    avatar: 'https://ui-avatars.com/api/?name=农业科技达人&background=random',
    content: '使用AI平台的农业监测功能，成功预测了农田的病虫害情况，帮助农民减少了损失。',
    likes: 256,
    comments: 48,
    time: '4小时前',
    tags: ['农业', '病虫害', '预测']
  },
  {
    id: 3,
    user: '区块链开发者',
    avatar: 'https://ui-avatars.com/api/?name=区块链开发者&background=random',
    content: 'AI平台的区块链集成功能非常强大，可以实现数据的安全存储和共享。',
    likes: 192,
    comments: 27,
    time: '6小时前',
    tags: ['区块链', '数据安全', '集成']
  }
]

export function Community() {
  const [liveStreams, setLiveStreams] = useState<LiveStream[]>(initialLiveStreams)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedStream, setSelectedStream] = useState<LiveStream | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [commentText, setCommentText] = useState('')
  const [comments, setComments] = useState([
    { id: 1, user: '用户A', content: '这个直播内容非常棒！学到了很多关于AI的知识，感谢主播分享！', time: '1小时前', likes: 15 },
    { id: 2, user: '用户B', content: '请问这个技术在实际生产环境中应用效果如何？', time: '30分钟前', likes: 8 },
    { id: 3, user: '用户C', content: '主播讲得很清晰，希望能有更多这样的直播！', time: '15分钟前', likes: 12 },
  ])
  const [likedPosts, setLikedPosts] = useState<{[key: number]: boolean}>({})
  const [postLikes, setPostLikes] = useState<{[key: number]: number}>({
    1: 128,
    2: 256,
    3: 192
  })
  
  // 从后端获取直播流数据
  useEffect(() => {
    const fetchLiveStreams = async () => {
      try {
        const response = await apiClient.getCommunityLiveStreams()
        if (response.success && response.data) {
          setLiveStreams(response.data)
          // 设置第一个直播流为默认选中
          if (response.data.length > 0) {
            setSelectedStream(response.data[0])
          }
        }
      } catch (error) {
        console.error('获取直播流失败:', error)
      }
    }
    
    fetchLiveStreams()
  }, [])
  
  // 搜索功能实现
  const filteredStreams = liveStreams.filter(stream => 
    stream.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    stream.streamer.toLowerCase().includes(searchQuery.toLowerCase()) ||
    stream.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  )
  
  const filteredPosts = communityPosts.filter(post => 
    post.user.toLowerCase().includes(searchQuery.toLowerCase()) ||
    post.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  )
  
  // 提交评论
  const handleSubmitComment = async () => {
    if (commentText.trim() && selectedStream) {
      try {
        const response = await apiClient.submitComment(selectedStream.id, commentText)
        if (response.success) {
          const newComment = {
            id: comments.length + 1,
            user: '我',
            content: commentText,
            time: '刚刚',
            likes: 0
          }
          setComments([newComment, ...comments])
          setCommentText('')
        }
      } catch (error) {
        console.error('提交评论失败:', error)
      }
    }
  }
  
  // 点赞功能
  const handleLikePost = async (postId: number) => {
    try {
      const response = await apiClient.likePost(postId)
      if (response.success) {
        setLikedPosts(prev => ({
          ...prev,
          [postId]: !prev[postId]
        }))
        setPostLikes(prev => ({
          ...prev,
          [postId]: prev[postId] + (likedPosts[postId] ? -1 : 1)
        }))
      }
    } catch (error) {
      console.error('点赞失败:', error)
    }
  }
  
  // 点赞评论
  const handleLikeComment = (commentId: number) => {
    setComments(prevComments => 
      prevComments.map(comment => 
        comment.id === commentId 
          ? { ...comment, likes: comment.likes + 1 } 
          : comment
      )
    )
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="bg-gradient-to-r from-tech-primary/10 to-tech-secondary/10 rounded-lg p-6 border border-tech-primary/20">
        <h1 className="text-3xl font-bold gradient-text mb-2">AI社区</h1>
        <p className="text-gray-300">参与直播、分享技术、交流经验</p>
      </div>

      {/* 搜索栏 */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
        <Input
          placeholder="搜索直播、帖子或用户..."
          className="pl-10 bg-tech-dark/50 border-tech-primary/20 text-white placeholder:text-gray-400"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      {/* 直播入口按钮 */}
      <div className="flex justify-center">
        <Button variant="tech" size="lg" className="flex items-center space-x-2">
          <Play className="w-5 h-5" />
          <span>进入直播</span>
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 直播区域 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 直播播放器 */}
          <Card className="glass-effect overflow-hidden">
            {selectedStream ? (
              <>
                <CardHeader className="bg-tech-dark/80 border-b border-tech-primary/20">
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Video className="w-5 h-5 text-tech-primary" />
                      <span>{selectedStream.title}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Button variant="tech" size="sm" className="flex items-center space-x-1">
                        <Users className="w-4 h-4" />
                        <span>{selectedStream.viewers}</span>
                      </Button>
                      <span className="text-xs px-2 py-1 bg-red-500 rounded-full animate-pulse">
                        直播中
                      </span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="relative bg-black w-full h-[400px] flex items-center justify-center">
                      {selectedStream.cover_image ? (
                        <img 
                          src={selectedStream.cover_image} 
                          alt={selectedStream.title} 
                          className="w-full h-full object-cover opacity-70"
                        />
                      ) : (
                        <div className="bg-gradient-to-br from-tech-primary to-tech-secondary w-full h-full flex items-center justify-center opacity-70">
                          <Video className="w-20 h-20 text-white/50" />
                        </div>
                      )}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Button 
                        variant="secondary" 
                        size="icon" 
                        className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/30"
                        onClick={() => setIsPlaying(!isPlaying)}
                      >
                        {isPlaying ? <Pause className="w-8 h-8" /> : <Play className="w-8 h-8 ml-1" />}
                      </Button>
                    </div>
                    {/* Video controls overlay */}
                    <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="text-white hover:bg-white/20"
                            onClick={() => setIsMuted(!isMuted)}
                          >
                            {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                          </Button>
                          <div className="text-sm text-white">
                            {selectedStream.streamer}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button variant="ghost" size="icon" className="text-white hover:bg-white/20">
                            <Heart className="w-5 h-5" />
                          </Button>
                          <Button variant="ghost" size="icon" className="text-white hover:bg-white/20">
                            <MessageSquare className="w-5 h-5" />
                          </Button>
                          <Button variant="ghost" size="icon" className="text-white hover:bg-white/20">
                            <Share2 className="w-5 h-5" />
                          </Button>
                          <Button variant="ghost" size="icon" className="text-white hover:bg-white/20">
                            <MoreVertical className="w-5 h-5" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </>
            ) : (
              <CardContent className="flex items-center justify-center h-[400px]">
                <p className="text-gray-400">正在加载直播...</p>
              </CardContent>
            )}
          </Card>

          {/* 直播评论区 */}
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <MessageSquare className="w-5 h-5 text-tech-primary" />
                <span>评论 ({Math.floor(Math.random() * 1000)})</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex space-x-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-tech-primary to-tech-secondary flex items-center justify-center font-bold">
                  我
                </div>
                <div className="flex-1">
                  <Input 
                    placeholder="输入评论..." 
                    className="bg-tech-dark/50 border-tech-primary/20 text-white placeholder:text-gray-400"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                  />
                  <div className="flex justify-end mt-2 space-x-2">
                    <Button variant="outline" size="sm" onClick={() => setCommentText('')}>取消</Button>
                    <Button variant="tech" size="sm" onClick={handleSubmitComment} disabled={!commentText.trim()}>发送</Button>
                  </div>
                </div>
              </div>
              {/* 评论列表 */}
              {comments.map((comment) => (
                <div key={comment.id} className="flex space-x-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-random-1 to-random-2 flex items-center justify-center font-bold">
                    {comment.user.charAt(0)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-sm font-medium">{comment.user}</span>
                      <span className="text-xs text-gray-400">{comment.time}</span>
                    </div>
                    <p className="text-sm text-gray-300">
                      {comment.content}
                    </p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                      <button 
                        className={`flex items-center space-x-1 ${comment.likes > 0 ? 'text-tech-primary' : 'hover:text-tech-primary'}`}
                        onClick={() => handleLikeComment(comment.id)}
                      >
                        <Heart className="w-3 h-3" />
                        <span>{comment.likes}</span>
                      </button>
                      <button className="hover:text-tech-primary">回复</button>
                      <button className="hover:text-tech-primary">举报</button>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* 侧边栏 */}
        <div className="space-y-6">
          {/* 直播列表 */}
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Video className="w-5 h-5 text-tech-primary" />
                <span>热门直播</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {filteredStreams.map((stream) => (
                <div 
                  key={stream.id}
                  className={`cursor-pointer rounded-lg overflow-hidden border ${selectedStream?.id === stream.id ? 'border-tech-primary' : 'border-tech-primary/10'} hover:border-tech-primary/50 transition-all duration-300`}
                  onClick={() => setSelectedStream(stream)}
                >
                  <div className="relative">
                    {stream.cover_image ? (
                      <img 
                        src={stream.cover_image} 
                        alt={stream.title} 
                        className="w-full h-32 object-cover"
                      />
                    ) : (
                      <div className="bg-gradient-to-br from-tech-primary to-tech-secondary w-full h-32 flex items-center justify-center">
                        <Video className="w-10 h-10 text-white/50" />
                      </div>
                    )}
                    <div className="absolute top-2 left-2 text-xs px-2 py-1 bg-red-500 rounded-full animate-pulse">
                      直播中
                    </div>
                    <div className="absolute top-2 right-2 text-xs px-2 py-1 bg-black/70 rounded-full text-white">
                      {stream.viewers}
                    </div>
                  </div>
                  <div className="p-2 bg-tech-dark/50">
                    <h4 className="font-medium text-sm line-clamp-1">{stream.title}</h4>
                    <p className="text-xs text-gray-400 mt-1">{stream.streamer}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* 社区推荐 */}
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Star className="w-5 h-5 text-tech-primary" />
                <span>社区推荐</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {filteredPosts.map((post) => (
                <div key={post.id} className="space-y-2">
                  <div className="flex items-center space-x-3">
                    <img 
                      src={post.avatar} 
                      alt={post.user} 
                      className="w-10 h-10 rounded-full"
                    />
                    <div>
                      <h4 className="text-sm font-medium">{post.user}</h4>
                      <p className="text-xs text-gray-400">{post.time}</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-300 line-clamp-3">{post.content}</p>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <div className="flex items-center space-x-4">
                      <button 
                        className={`flex items-center space-x-1 ${likedPosts[post.id] ? 'text-tech-primary' : 'hover:text-tech-primary'}`}
                        onClick={() => handleLikePost(post.id)}
                      >
                        <Heart className="w-3 h-3" />
                        <span>{postLikes[post.id]}</span>
                      </button>
                      <button className="flex items-center space-x-1 hover:text-tech-primary">
                        <MessageSquare className="w-3 h-3" />
                        <span>{post.comments}</span>
                      </button>
                      <button className="flex items-center space-x-1 hover:text-tech-primary">
                        <Share2 className="w-3 h-3" />
                      </button>
                    </div>
                    <div className="flex items-center space-x-1">
                      {post.tags.slice(0, 2).map((tag, index) => (
                        <span key={index} className="px-2 py-0.5 bg-tech-primary/20 rounded-full">
                          #{tag}
                        </span>
                      ))}
                      {post.tags.length > 2 && (
                        <span className="text-xs text-gray-500">+{post.tags.length - 2}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>


        </div>
      </div>
    </div>
  )
}

export default Community
