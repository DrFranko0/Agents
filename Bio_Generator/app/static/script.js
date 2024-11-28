document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        event.preventDefault();
        
        const career = document.querySelector('#career').value;
        const personality = document.querySelector('#personality').value;
        const interests = document.querySelector('#interests').value.split(',').map(interest => interest.trim());
        const relationship = document.querySelector('#relationship').value;

        const formData = new FormData();
        formData.append('career', career);
        formData.append('personality', personality);
        formData.append('interests', interests.join(','));
        formData.append('relationship', relationship);

        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data && data.bio) {
                document.querySelector('#result-container').innerHTML = `
                    <h1>Your Personalized Bio</h1>
                    <p>${data.bio}</p>
                `;
            }
        })
        .catch(error => console.error('Error generating bio:', error));
    });
});
